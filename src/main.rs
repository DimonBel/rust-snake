use actix_web::{web, App, HttpResponse, HttpServer};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Game {
    id: String,
}

#[derive(Deserialize)]
struct Board {
    height: i32,
    width: i32,
    food: Vec<Coord>,
    snakes: Vec<Snake>,
}

#[derive(Deserialize)]
struct Coord {
    x: i32,
    y: i32,
}

#[derive(Deserialize)]
struct Snake {
    id: String,
    body: Vec<Coord>,
    health: i32,
}

#[derive(Deserialize)]
struct GameState {
    game: Game,
    turn: i32,
    board: Board,
    you: Snake,
}

#[derive(Serialize)]
struct MoveResponse {
    r#move: String,
}

#[derive(Serialize)]
struct StartResponse {
    color: String,
}

// Define strategy space for bilinear duel (simplified to 2D for movement directions)
fn bilinear_duel(state: &GameState) -> String {
    let you = &state.you;
    let head = &you.body[0];
    let board = &state.board;

    // Define strategy space K for player (snake): probabilities for moves (up, down, left, right)
    let n = 4; // Four possible moves
    let moves = vec!["up", "down", "left", "right"];
    let mut strategy = DVector::from_element(n, 0.25); // Initial uniform strategy

    // Define payoff matrix M (heuristic-based, 4x4 for simplicity)
    let mut M = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            M[(i, j)] = compute_payoff(&moves[i], &moves[j], head, board);
        }
    }

    // Define constraints for K (simplex: sum of probabilities = 1, non-negative)
    let m = 1; // One constraint for sum = 1
    let w = DVector::from_element(n, 1.0); // Weights for sum constraint
    let b: f64 = 1.0; // Explicitly type b as f64 to resolve ambiguity

    // Opponent strategy space K0 (simplified, assume uniform for now)
    let m0 = 1;
    let w0 = DVector::from_element(n, 1.0);
    let b0 = 1.0;

    // Solve linear program: max sum(lambda_i * b0_i) s.t. x in K and x^T M = sum(lambda_i * w0_i)
    let lambda = DVector::from_element(m0, 1.0 / m0 as f64); // Simplified lambda
    let mut x = DVector::zeros(n);
    for i in 0..n {
        x[i] = if strategy.iter().all(|&p| p >= 0.0) && (strategy.dot(&w) - b).abs() < 1e-6 {
            strategy[i]
        } else {
            0.25 // Fallback to uniform
        };
    }

    // Choose move with highest probability
    let max_idx = x
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    moves[max_idx].to_string()
}

// Add these after your existing struct definitions
#[derive(Clone, Eq, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn from_coord(coord: &Coord) -> Self {
        Point {
            x: coord.x,
            y: coord.y,
        }
    }
}

// Add this new function
fn flood_fill(board: &Board, start: &Coord, visited: &mut Vec<Point>) -> i32 {
    let mut stack = vec![Point::from_coord(start)];
    let mut space_count = 0;
    let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    while let Some(current) = stack.pop() {
        if visited.contains(&current) {
            continue;
        }

        // Check if position is valid
        if current.x < 0
            || current.x >= board.width
            || current.y < 0
            || current.y >= board.height
        {
            continue;
        }

        // Check for collision with snake bodies
        let is_snake = board.snakes.iter().any(|snake| {
            snake.body.iter().any(|segment| {
                segment.x == current.x && segment.y == current.y
            })
        });

        if is_snake {
            continue;
        }

        visited.push(current.clone());
        space_count += 1;

        // Add adjacent cells to stack
        for (dx, dy) in directions.iter() {
            stack.push(Point {
                x: current.x + dx,
                y: current.y + dy,
            });
        }
    }

    space_count
}

// Modify your compute_payoff function to use flood fill
fn compute_payoff(my_move: &str, opp_move: &str, head: &Coord, board: &Board) -> f64 {
    let new_pos = match my_move {
        "up" => Coord {
            x: head.x,
            y: head.y + 1,
        },
        "down" => Coord {
            x: head.x,
            y: head.y - 1,
        },
        "left" => Coord {
            x: head.x - 1,
            y: head.y,
        },
        "right" => Coord {
            x: head.x + 1,
            y: head.y,
        },
        _ => return 0.0,
    };

    // Basic collision checks
    if new_pos.x < 0 || new_pos.x >= board.width || new_pos.y < 0 || new_pos.y >= board.height {
        return -10.0;
    }

    for snake in &board.snakes {
        for segment in &snake.body {
            if new_pos.x == segment.x && new_pos.y == segment.y {
                return -10.0;
            }
        }
    }

    // Perform flood fill from the new position
    let mut visited = Vec::new();
    let available_space = flood_fill(board, &new_pos, &mut visited);

    // Get distance to nearest food
    let food_dist = board
        .food
        .iter()
        .map(|f| ((f.x - new_pos.x).abs() + (f.y - new_pos.y).abs()) as f64)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(100.0);

    // Calculate payoff based on available space and food distance
    let space_score = available_space as f64 * 2.0;
    let food_score = 10.0 - food_dist;

    // Combined score
    let base_payoff = space_score + food_score;

    // Reduce payoff if opponent might move to same space
    if my_move == opp_move {
        base_payoff * 0.5
    } else {
        base_payoff
    }
}

// API endpoints
async fn index() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "api_version": "1",
        "author": "xAI",
        "color": "#FF0000",
        "head": "default",
        "tail": "default",
    }))
}

async fn start(_state: web::Json<GameState>) -> HttpResponse {
    HttpResponse::Ok().json(StartResponse {
        color: "#FF0000".to_string(),
    })
}

async fn r#move(state: web::Json<GameState>) -> HttpResponse {
    let chosen_move = bilinear_duel(&state);
    HttpResponse::Ok().json(MoveResponse {
        r#move: chosen_move,
    })
}

async fn end(_state: web::Json<GameState>) -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
            .route("/start", web::post().to(start))
            .route("/move", web::post().to(r#move))
            .route("/end", web::post().to(end))
    })
    .bind("0.0.0.0:8000")?
    .run()
    .await
}
