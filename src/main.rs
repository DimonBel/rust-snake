use log::{debug, info, warn};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use warp::Filter;

// -------------------- Battlesnake API Structs --------------------
#[derive(Deserialize, Serialize, Debug, Clone)]
struct Coord {
    x: i32,
    y: i32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Battlesnake {
    id: String,
    name: String,
    health: i32,
    body: Vec<Coord>,
    latency: String,
    head: Coord,
    length: i32,
    shout: String,
    squad: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Board {
    height: i32,
    width: i32,
    food: Vec<Coord>,
    hazards: Vec<Coord>,
    snakes: Vec<Battlesnake>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Game {
    id: String,
    ruleset: HashMap<String, serde_json::Value>,
    map: String,
    timeout: i32,
    source: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct GameRequest {
    game: Game,
    turn: i32,
    board: Board,
    you: Battlesnake,
}

#[derive(Serialize)]
struct GameResponse {
    #[serde(rename = "move")]
    direction: String,
    shout: Option<String>,
}

#[derive(Serialize)]
struct InfoResponse {
    apiversion: String,
    author: String,
    color: String,
    head: String,
    tail: String,
}

// -------------------- Flood Fill Safety Check --------------------
struct FloodFillChecker {
    board: Vec<Vec<bool>>,
    width: usize,
    height: usize,
}

impl FloodFillChecker {
    fn new(game_state: &GameRequest) -> Self {
        let width = game_state.board.width as usize;
        let height = game_state.board.height as usize;
        let mut board = vec![vec![false; width]; height];

        for snake in &game_state.board.snakes {
            for part in &snake.body {
                if part.x >= 0 && part.x < width as i32 && part.y >= 0 && part.y < height as i32 {
                    board[part.y as usize][part.x as usize] = true;
                }
            }
        }

        for hazard in &game_state.board.hazards {
            if hazard.x >= 0 && hazard.x < width as i32 && hazard.y >= 0 && hazard.y < height as i32
            {
                board[hazard.y as usize][hazard.x as usize] = true;
            }
        }

        FloodFillChecker {
            board,
            width,
            height,
        }
    }

    fn flood_fill(&self, start: Coord) -> usize {
        if start.x < 0
            || start.x >= self.width as i32
            || start.y < 0
            || start.y >= self.height as i32
            || self.board[start.y as usize][start.x as usize]
        {
            return 0;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

        while let Some(coord) = queue.pop_front() {
            if visited.contains(&(coord.x, coord.y)) {
                continue;
            }
            visited.insert((coord.x, coord.y));

            for (dx, dy) in directions.iter() {
                let new_x = coord.x + dx;
                let new_y = coord.y + dy;
                if new_x >= 0
                    && new_x < self.width as i32
                    && new_y >= 0
                    && new_y < self.height as i32
                {
                    if !self.board[new_y as usize][new_x as usize]
                        && !visited.contains(&(new_x, new_y))
                    {
                        queue.push_back(Coord { x: new_x, y: new_y });
                    }
                }
            }
        }

        visited.len()
    }

    fn get_safe_moves(&self, head: &Coord) -> Vec<(String, usize)> {
        let moves = vec![
            (
                "up",
                Coord {
                    x: head.x,
                    y: head.y + 1,
                },
            ),
            (
                "down",
                Coord {
                    x: head.x,
                    y: head.y - 1,
                },
            ),
            (
                "left",
                Coord {
                    x: head.x - 1,
                    y: head.y,
                },
            ),
            (
                "right",
                Coord {
                    x: head.x + 1,
                    y: head.y,
                },
            ),
        ];

        moves
            .into_iter()
            .map(|(dir, coord)| (dir.to_string(), self.flood_fill(coord)))
            .collect()
    }
}

// -------------------- Bilinear Duel --------------------
struct BilinearDuel<'a> {
    game_state: &'a GameRequest,
}

impl<'a> BilinearDuel<'a> {
    fn new(game_state: &'a GameRequest) -> Self {
        BilinearDuel { game_state }
    }

    fn create_payoff_matrix(
        &self,
        my_snake: &Battlesnake,
        opp_snake: &Battlesnake,
    ) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(4, 4);
        let moves = vec![
            Coord { x: 0, y: 1 },
            Coord { x: 0, y: -1 },
            Coord { x: -1, y: 0 },
            Coord { x: 1, y: 0 },
        ];
        for (i, my_move) in moves.iter().enumerate() {
            for (j, opp_move) in moves.iter().enumerate() {
                let my_pos = Coord {
                    x: my_snake.head.x + my_move.x,
                    y: my_snake.head.y + my_move.y,
                };
                let opp_pos = Coord {
                    x: opp_snake.head.x + opp_move.x,
                    y: opp_snake.head.y + opp_move.y,
                };
                matrix[(i, j)] = self.calculate_payoff(&my_pos, &opp_pos, my_snake, opp_snake);
            }
        }
        matrix
    }

    fn calculate_payoff(
        &self,
        my_pos: &Coord,
        opp_pos: &Coord,
        my_snake: &Battlesnake,
        opp_snake: &Battlesnake,
    ) -> f64 {
        let mut payoff = 0.0;
        let w = self.game_state.board.width;
        let h = self.game_state.board.height;

        if my_pos.x < 0 || my_pos.x >= w || my_pos.y < 0 || my_pos.y >= h {
            payoff -= 1000.0;
        }
        if opp_pos.x < 0 || opp_pos.x >= w || opp_pos.y < 0 || opp_pos.y >= h {
            payoff += 1000.0;
        }

        for part in &my_snake.body[..my_snake.body.len().saturating_sub(1)] {
            if my_pos.x == part.x && my_pos.y == part.y {
                payoff -= 1000.0;
            }
        }
        for part in &opp_snake.body[..opp_snake.body.len().saturating_sub(1)] {
            if my_pos.x == part.x && my_pos.y == part.y {
                payoff -= 500.0;
            }
            if opp_pos.x == part.x && opp_pos.y == part.y {
                payoff += 500.0;
            }
        }

        if my_pos.x == opp_pos.x && my_pos.y == opp_pos.y {
            if my_snake.length >= opp_snake.length {
                payoff += 100.0;
            } else {
                payoff -= 100.0;
            }
        }

        for food in &self.game_state.board.food {
            let my_dist = (my_pos.x - food.x).abs() + (my_pos.y - food.y).abs();
            let opp_dist = (opp_pos.x - food.x).abs() + (opp_pos.y - food.y).abs();
            if my_dist < opp_dist {
                payoff += 10.0 / (my_dist as f64 + 1.0);
            } else {
                payoff -= 5.0 / (opp_dist as f64 + 1.0);
            }
        }

        if my_snake.health < opp_snake.health {
            payoff += 5.0;
        }
        payoff
    }

    fn solve_bilinear_duel(&self) -> String {
        if self.game_state.board.snakes.len() != 2 {
            warn!("Bilinear duel only valid for 2 snakes");
            return "up".to_string();
        }
        let my_snake = &self.game_state.you;
        let opp_snake = self
            .game_state
            .board
            .snakes
            .iter()
            .find(|s| s.id != my_snake.id)
            .unwrap();
        let matrix = self.create_payoff_matrix(my_snake, opp_snake);
        debug!("Payoff matrix: {}", matrix);

        let strategy = self.minimax_strategy(&matrix);
        let moves = vec!["up", "down", "left", "right"];
        let best_idx = strategy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        moves[best_idx].to_string()
    }

    fn minimax_strategy(&self, matrix: &DMatrix<f64>) -> Vec<f64> {
        let n = matrix.nrows();
        let mut strat = vec![0.0; n];
        for i in 0..n {
            strat[i] = matrix.row(i).sum() / matrix.ncols() as f64;
        }
        let sum: f64 = strat.iter().sum();
        if sum > 0.0 {
            for v in strat.iter_mut() {
                *v /= sum;
            }
        } else {
            for v in strat.iter_mut() {
                *v = 1.0 / n as f64;
            }
        }
        strat
    }
}

// -------------------- Main Move Logic --------------------
fn get_move(game_state: &GameRequest) -> String {
    match game_state.board.snakes.len() {
        0..=1 => {
            info!("Single snake - basic survival");
            let flood = FloodFillChecker::new(game_state);
            let safe_moves = flood.get_safe_moves(&game_state.you.head);
            safe_moves
                .iter()
                .max_by_key(|(_, s)| *s)
                .map(|(d, _)| d.clone())
                .unwrap_or("up".to_string())
        }
        2 => {
            info!("Two snakes - bilinear duel");
            let duel = BilinearDuel::new(game_state);
            duel.solve_bilinear_duel()
        }
        _ => {
            info!("3+ snakes - flood fill algorithm");
            let flood = FloodFillChecker::new(game_state);
            let safe_moves = flood.get_safe_moves(&game_state.you.head);
            safe_moves
                .iter()
                .max_by_key(|(_, s)| *s)
                .map(|(d, _)| d.clone())
                .unwrap_or("up".to_string())
        }
    }
}

// -------------------- HTTP Handlers --------------------
async fn handle_index() -> Result<impl warp::Reply, Infallible> {
    let info = InfoResponse {
        apiversion: "1".to_string(),
        author: "Dumas".to_string(),
        color: "#fdb0ceff".to_string(),
        head: "default".to_string(),
        tail: "default".to_string(),
    };
    Ok(warp::reply::json(&info))
}

async fn handle_start(_req: GameRequest) -> Result<impl warp::Reply, Infallible> {
    info!("Game started");
    Ok(warp::reply::with_status("", warp::http::StatusCode::OK))
}

async fn handle_move(req: GameRequest) -> Result<impl warp::Reply, Infallible> {
    let mv = get_move(&req);
    info!("Move chosen: {}", mv);
    let res = GameResponse {
        direction: mv.clone(),
        shout: Some(format!("Moving {}", mv)),
    };
    Ok(warp::reply::json(&res))
}

async fn handle_end(_req: GameRequest) -> Result<impl warp::Reply, Infallible> {
    info!("Game ended");
    Ok(warp::reply::with_status("", warp::http::StatusCode::OK))
}

// -------------------- Server Entry --------------------
#[tokio::main]
async fn main() {
    env_logger::init();
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse::<u16>()
        .unwrap();
    info!("Starting server on port {}", port);

    let index = warp::get().and(warp::path::end()).and_then(handle_index);
    let start = warp::post()
        .and(warp::path("start"))
        .and(warp::body::json())
        .and_then(handle_start);
    let mv = warp::post()
        .and(warp::path("move"))
        .and(warp::body::json())
        .and_then(handle_move);
    let end = warp::post()
        .and(warp::path("end"))
        .and(warp::body::json())
        .and_then(handle_end);

    let routes = index.or(start).or(mv).or(end).with(
        warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"])
            .allow_methods(vec!["GET", "POST"]),
    );

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}
