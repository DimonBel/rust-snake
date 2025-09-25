use actix_web::{web, App, HttpResponse, HttpServer};
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

#[derive(Deserialize, Clone)]  // Add Clone here
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

// Move this near other struct definitions
#[derive(Clone, Debug)]
struct Move {
    direction: String,
    score: f64,
}

impl Move {
    fn new(direction: &str) -> Self {
        Move {
            direction: direction.to_string(),
            score: 0.0,
        }
    }
}

// Add this helper function early in the file
fn get_new_position(head: &Coord, direction: &str) -> Coord {
    match direction {
        "up" => Coord { x: head.x, y: head.y + 1 },
        "down" => Coord { x: head.x, y: head.y - 1 },
        "left" => Coord { x: head.x - 1, y: head.y },
        "right" => Coord { x: head.x + 1, y: head.y },
        _ => Coord { x: head.x, y: head.y },
    }
}

fn manhattan_distance(a: &Coord, b: &Coord) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

fn is_safe_move(pos: &Coord, board: &Board, snake_length: usize) -> bool {
    // Check board boundaries
    if pos.x < 0 || pos.x >= board.width || pos.y < 0 || pos.y >= board.height {
        return false;
    }

    // Check snake collisions
    for snake in &board.snakes {
        for (i, segment) in snake.body.iter().enumerate() {
            if pos.x == segment.x && pos.y == segment.y {
                // Allow moving to tail position if it's going to move
                if !(i == snake.body.len() - 1 && snake.body.len() == snake_length) {
                    return false;
                }
            }
        }
    }

    true
}

// Update evaluate_food to be more efficient and actually use health parameter
fn evaluate_food(pos: &Coord, board: &Board, health: i32) -> Option<(f64, String)> {
    let mut nearest_food = None;
    let mut min_dist = f64::MAX;

    for food in &board.food {
        let dist = manhattan_distance(pos, food) as f64;
        if dist < min_dist {
            min_dist = dist;
            let dir = if (food.x - pos.x).abs() > (food.y - pos.y).abs() {
                if food.x > pos.x { "right" } else { "left" }
            } else {
                if food.y > pos.y { "up" } else { "down" }
            }.to_string();
            nearest_food = Some((dist, dir));
        }
    }

    // Adjust score based on health
    nearest_food.map(|(dist, dir)| {
        let urgency = if health < 25 { 1.5 } else { 1.0 };
        (dist * urgency, dir)
    })
}

fn calculate_food_score(distance: f64, health: i32) -> f64 {
    let base_score = 100.0 - distance;
    
    // Increase urgency when health is low
    if health < 25 {
        base_score * 3.0
    } else if health < 50 {
        base_score * 1.5
    } else {
        base_score
    }
}

fn evaluate_threats(pos: &Coord, board: &Board, you: &Snake) -> f64 {
    let mut threat_score = 0.0;

    for snake in &board.snakes {
        if snake.id != you.id {
            let head_dist = manhattan_distance(pos, &snake.body[0]);
            
            // Evaluate head-to-head scenarios
            if head_dist <= 2 {
                if you.body.len() <= snake.body.len() {
                    threat_score -= 150.0; // Strong penalty for risky head-to-head
                } else {
                    threat_score += 50.0; // Potential to eliminate shorter snake
                }
            }
        }
    }

    threat_score
}

fn evaluate_center_control(pos: &Coord, board: &Board) -> f64 {
    let center_x = board.width as f64 / 2.0;
    let center_y = board.height as f64 / 2.0;
    let dist_from_center = ((pos.x as f64 - center_x).powi(2) + 
                           (pos.y as f64 - center_y).powi(2)).sqrt();
    
    // Prefer positions closer to center
    25.0 - dist_from_center * 2.0
}

// Define strategy space for bilinear duel (simplified to 2D for movement directions)
fn bilinear_duel(state: &GameState) -> String {
    let you = &state.you;
    let head = &you.body[0];
    let board = &state.board;
    
    let possible_moves = vec![
        Move::new("up"),
        Move::new("down"),
        Move::new("left"),
        Move::new("right"),
    ];

    // Find best move using weighted scoring
    let best_move = evaluate_moves(possible_moves, head, you, board);
    best_move.direction
}

fn evaluate_moves(mut moves: Vec<Move>, head: &Coord, you: &Snake, board: &Board) -> Move {
    for move_option in &mut moves {
        let new_pos = get_new_position(head, &move_option.direction);
        
        // Initialize score
        let mut score = 0.0;
        
        // Immediate death check
        if !is_safe_move(&new_pos, board, you.body.len()) {
            move_option.score = f64::NEG_INFINITY;
            continue;
        }

        // Space evaluation (weighted highest)
        let mut visited = Vec::new();
        let available_space = flood_fill(board, &new_pos, &mut visited);
        score += available_space as f64 * 5.0; // High weight for available space

        // Food evaluation
        if let Some((food_dist, food_dir)) = evaluate_food(&new_pos, board, you.health) {
            let food_score = calculate_food_score(food_dist, you.health);
            if move_option.direction == food_dir {
                score += food_score;
            }
        }

        // Threat evaluation
        score += evaluate_threats(&new_pos, board, you);

        // Center control evaluation
        score += evaluate_center_control(&new_pos, board);

        move_option.score = score;
    }

    // Sort by score and return best move
    moves.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    moves.into_iter().next().unwrap_or(Move::new("up"))
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

// Добавьте эту функцию для проверки безопасности хода
fn is_move_safe(new_pos: &Coord, board: &Board, snake_length: usize) -> bool {
    // Проверка на выход за пределы поля
    if new_pos.x < 0 || new_pos.x >= board.width || new_pos.y < 0 || new_pos.y >= board.height {
        return false;
    }

    // Проверка столкновений со змеями
    for snake in &board.snakes {
        for (i, segment) in snake.body.iter().enumerate() {
            // Пропускаем последний сегмент хвоста, так как он движется
            if i == snake.body.len() - 1 && snake.body.len() == snake_length {
                continue;
            }
            if new_pos.x == segment.x && new_pos.y == segment.y {
                return false;
            }
        }
    }

    true
}

// Обновленная функция compute_payoff
fn compute_payoff(my_move: &str, _opp_move: &str, head: &Coord, board: &Board) -> f64 {
    let new_pos = match my_move {
        "up" => Coord { x: head.x, y: head.y + 1 },
        "down" => Coord { x: head.x, y: head.y - 1 },
        "left" => Coord { x: head.x - 1, y: head.y },
        "right" => Coord { x: head.x + 1, y: head.y },
        _ => return -100.0,
    };

    // Check for immediate death
    if !is_move_safe(&new_pos, board, board.snakes[0].body.len()) {
        return -100.0;
    }

    let mut score = 0.0;

    // Space evaluation
    let mut visited = Vec::new();
    let available_space = flood_fill(board, &new_pos, &mut visited);
    score += available_space as f64 * 5.0;

    // Food evaluation
    if let Some((food_distance, _)) = evaluate_food(&new_pos, board, board.snakes[0].health) {
        score += 100.0 - food_distance as f64;  // Removed unnecessary parentheses
    }

    score
}

// API endpoints
async fn index() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "api_version": "1",
        "author": "Dumas",
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
