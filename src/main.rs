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

// Add these strategy-related structs and implementations
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
        if !is_safe_move(&new_pos, board) {
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
fn compute_payoff(my_move: &str, opp_move: &str, head: &Coord, board: &Board) -> f64 {
    let new_pos = match my_move {
        "up" => Coord { x: head.x, y: head.y + 1 },
        "down" => Coord { x: head.x, y: head.y - 1 },
        "left" => Coord { x: head.x - 1, y: head.y },
        "right" => Coord { x: head.x + 1, y: head.y },
        _ => return -100.0,
    };

    // Проверяем безопасность хода
    if !is_move_safe(&new_pos, board, board.snakes[0].body.len()) {
        return -100.0;
    }

    let mut score = 0.0;

    // Выполняем flood fill
    let mut visited = Vec::new();
    let available_space = flood_fill(board, &new_pos, &mut visited);
    
    // Оценка доступного пространства (важнее всего)
    score += available_space as f64 * 3.0;

    // Избегаем краев поля (если не гонимся за едой)
    if new_pos.x <= 1 || new_pos.x >= board.width - 2 || 
       new_pos.y <= 1 || new_pos.y >= board.height - 2 {
        score -= 15.0;
    }

    // Оценка расстояния до еды
    if let Some(nearest_food) = find_nearest_food(&new_pos, board) {
        let food_distance = manhattan_distance(&new_pos, &nearest_food);
        // Если здоровье низкое, увеличиваем важность еды
        if board.snakes[0].health < 50 {
            score += (100.0 - food_distance as f64) * 2.0;
        } else {
            score += (100.0 - food_distance as f64);
        }
    }

    // Избегаем ходов, которые могут привести к столкновению с другими змеями
    for snake in &board.snakes {
        if snake.id != board.snakes[0].id {  // не наша змея
            let snake_head = &snake.body[0];
            let distance = manhattan_distance(&new_pos, snake_head);
            if distance <= 2 {
                // Если мы короче другой змеи, избегаем конфронтации
                if board.snakes[0].body.len() <= snake.body.len() {
                    score -= 50.0;
                }
            }
        }
    }

    score
}

// Вспомогательные функции
fn manhattan_distance(a: &Coord, b: &Coord) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

fn find_nearest_food(pos: &Coord, board: &Board) -> Option<Coord> {
    board.food.iter()
        .min_by_key(|food| manhattan_distance(pos, food))
        .cloned()
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
