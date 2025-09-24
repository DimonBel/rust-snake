use log::{debug, info, warn};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use warp::Filter;

// Battlesnake API structures
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

#[derive(Deserialize, Serialize, Debug)]
struct Board {
    height: i32,
    width: i32,
    food: Vec<Coord>,
    hazards: Vec<Coord>,
    snakes: Vec<Battlesnake>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Game {
    id: String,
    ruleset: HashMap<String, serde_json::Value>,
    map: String,
    timeout: i32,
    source: String,
}

#[derive(Deserialize, Serialize, Debug)]
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

// Algorithm implementations
struct FloodFillChecker {
    board: Vec<Vec<bool>>, // true = occupied/blocked
    width: usize,
    height: usize,
}

impl FloodFillChecker {
    fn new(game_state: &GameRequest) -> Self {
        let width = game_state.board.width as usize;
        let height = game_state.board.height as usize;
        let mut board = vec![vec![false; width]; height];

        // Mark snake bodies as occupied
        for snake in &game_state.board.snakes {
            for body_part in &snake.body {
                if body_part.x >= 0
                    && body_part.x < width as i32
                    && body_part.y >= 0
                    && body_part.y < height as i32
                {
                    board[body_part.y as usize][body_part.x as usize] = true;
                }
            }
        }

        // Mark hazards as occupied
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
                    && !self.board[new_y as usize][new_x as usize]
                    && !visited.contains(&(new_x, new_y))
                {
                    queue.push_back(Coord { x: new_x, y: new_y });
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
            .map(|(direction, coord)| {
                let space = self.flood_fill(coord);
                (direction.to_string(), space)
            })
            .collect()
    }
}

struct BilinearDuel {
    game_state: GameRequest,
}

impl BilinearDuel {
    fn new(game_state: GameRequest) -> Self {
        BilinearDuel { game_state }
    }

    fn create_payoff_matrix(&self, my_snake: &Battlesnake, opponent: &Battlesnake) -> DMatrix<f64> {
        // Create a 4x4 payoff matrix for the 4 possible moves
        // Rows: our moves (up, down, left, right)
        // Cols: opponent moves (up, down, left, right)
        let mut matrix = DMatrix::zeros(4, 4);

        let my_head = &my_snake.head;
        let opp_head = &opponent.head;

        let moves = vec![
            Coord { x: 0, y: 1 },  // up
            Coord { x: 0, y: -1 }, // down
            Coord { x: -1, y: 0 }, // left
            Coord { x: 1, y: 0 },  // right
        ];

        for (i, my_move) in moves.iter().enumerate() {
            for (j, opp_move) in moves.iter().enumerate() {
                let my_new_pos = Coord {
                    x: my_head.x + my_move.x,
                    y: my_head.y + my_move.y,
                };
                let opp_new_pos = Coord {
                    x: opp_head.x + opp_move.x,
                    y: opp_head.y + opp_move.y,
                };

                let payoff = self.calculate_payoff(&my_new_pos, &opp_new_pos, my_snake, opponent);
                matrix[(i, j)] = payoff;
            }
        }

        matrix
    }

    fn calculate_payoff(
        &self,
        my_pos: &Coord,
        opp_pos: &Coord,
        my_snake: &Battlesnake,
        opponent: &Battlesnake,
    ) -> f64 {
        let mut payoff = 0.0;

        let board_width = self.game_state.board.width;
        let board_height = self.game_state.board.height;

        // Boundary collision check
        if my_pos.x < 0 || my_pos.x >= board_width || my_pos.y < 0 || my_pos.y >= board_height {
            payoff -= 1000.0;
        }
        if opp_pos.x < 0 || opp_pos.x >= board_width || opp_pos.y < 0 || opp_pos.y >= board_height {
            payoff += 1000.0;
        }

        // Body collision check
        for body_part in &my_snake.body[..my_snake.body.len() - 1] {
            if my_pos.x == body_part.x && my_pos.y == body_part.y {
                payoff -= 1000.0;
            }
        }
        for body_part in &opponent.body[..opponent.body.len() - 1] {
            if my_pos.x == body_part.x && my_pos.y == body_part.y {
                payoff -= 500.0;
            }
            if opp_pos.x == body_part.x && opp_pos.y == body_part.y {
                payoff += 500.0;
            }
        }

        // Head-to-head collision
        if my_pos.x == opp_pos.x && my_pos.y == opp_pos.y {
            if my_snake.length >= opponent.length {
                payoff += 100.0;
            } else {
                payoff -= 100.0;
            }
        }

        // Distance to food
        for food in &self.game_state.board.food {
            let my_distance = (my_pos.x - food.x).abs() + (my_pos.y - food.y).abs();
            let opp_distance = (opp_pos.x - food.x).abs() + (opp_pos.y - food.y).abs();

            if my_distance < opp_distance {
                payoff += 10.0 / (my_distance as f64 + 1.0);
            } else {
                payoff -= 5.0 / (opp_distance as f64 + 1.0);
            }
        }

        // Health consideration
        if my_snake.health < opponent.health {
            // Prioritize food when low on health
            payoff += 5.0;
        }

        payoff
    }

    fn solve_bilinear_duel(&self) -> String {
        if self.game_state.board.snakes.len() != 2 {
            warn!("Bilinear duel should only be used with 2 snakes");
            return "up".to_string();
        }

        let my_snake = &self.game_state.you;
        let opponent = self
            .game_state
            .board
            .snakes
            .iter()
            .find(|s| s.id != my_snake.id)
            .unwrap();

        let payoff_matrix = self.create_payoff_matrix(my_snake, opponent);
        debug!("Payoff matrix: {}", payoff_matrix);

        // Solve using minimax with linear programming approximation
        let optimal_strategy = self.minimax_strategy(&payoff_matrix);

        // Convert strategy to move
        let moves = vec!["up", "down", "left", "right"];
        let best_move_idx = optimal_strategy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        moves[best_move_idx].to_string()
    }

    fn minimax_strategy(&self, payoff_matrix: &DMatrix<f64>) -> Vec<f64> {
        // Simplified minimax strategy calculation
        // In a full implementation, this would use linear programming
        let num_moves = payoff_matrix.nrows();
        let mut strategy = vec![0.0; num_moves];

        // Calculate expected payoff for each of our moves against uniform opponent strategy
        for i in 0..num_moves {
            let mut expected_payoff = 0.0;
            for j in 0..payoff_matrix.ncols() {
                expected_payoff += payoff_matrix[(i, j)];
            }
            strategy[i] = expected_payoff / payoff_matrix.ncols() as f64;
        }

        // Normalize to create a probability distribution
        let sum: f64 = strategy.iter().sum();
        if sum > 0.0 {
            for val in strategy.iter_mut() {
                *val /= sum;
            }
        } else {
            // Uniform distribution if all moves are bad
            for val in strategy.iter_mut() {
                *val = 1.0 / num_moves as f64;
            }
        }

        strategy
    }
}

// Main game logic
fn get_move(game_state: GameRequest) -> String {
    let num_snakes = game_state.board.snakes.len();

    if num_snakes >= 3 {
        // Use flood fill algorithm for 3+ snakes
        info!("Using flood fill algorithm with {} snakes", num_snakes);
        let flood_fill = FloodFillChecker::new(&game_state);
        let safe_moves = flood_fill.get_safe_moves(&game_state.you.head);

        // Choose the move with the most available space
        let best_move = safe_moves
            .iter()
            .max_by_key(|(_, space)| *space)
            .map(|(direction, space)| {
                info!("Move {} has {} available spaces", direction, space);
                direction.clone()
            })
            .unwrap_or_else(|| "up".to_string());

        best_move
    } else if num_snakes == 2 {
        // Use bilinear duel algorithm for 2 snakes
        info!("Using bilinear duel algorithm with 2 snakes");
        let bilinear_duel = BilinearDuel::new(game_state);
        bilinear_duel.solve_bilinear_duel()
    } else {
        // Single snake - just avoid walls and hazards
        info!("Single snake - using basic survival");
        let flood_fill = FloodFillChecker::new(&game_state);
        let safe_moves = flood_fill.get_safe_moves(&game_state.you.head);

        safe_moves
            .iter()
            .max_by_key(|(_, space)| *space)
            .map(|(direction, _)| direction.clone())
            .unwrap_or_else(|| "up".to_string())
    }
}

// HTTP handlers
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

async fn handle_start(game_request: GameRequest) -> Result<impl warp::Reply, Infallible> {
    info!("Game {} started", game_request.game.id);
    Ok(warp::reply::with_status("", warp::http::StatusCode::OK))
}

async fn handle_move(game_request: GameRequest) -> Result<impl warp::Reply, Infallible> {
    let game_id = game_request.game.id.clone();
    let turn = game_request.turn;

    info!("Turn {} for game {}", turn, game_id);

    let chosen_move = get_move(game_request);

    let response = GameResponse {
        direction: chosen_move.clone(),
        shout: Some(format!("Moving {}", chosen_move)),
    };

    info!("Chose move: {}", chosen_move);
    Ok(warp::reply::json(&response))
}

async fn handle_end(game_request: GameRequest) -> Result<impl warp::Reply, Infallible> {
    info!("Game {} ended", game_request.game.id);
    Ok(warp::reply::with_status("", warp::http::StatusCode::OK))
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid number");

    info!("Starting Battlesnake server on port {}", port);

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
