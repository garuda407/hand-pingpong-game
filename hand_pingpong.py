import cv2
import mediapipe as mp
import pygame
import numpy as np
import random

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Ping Pong")
clock = pygame.time.Clock()

# Colors
RED = (255, 0, 0)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
BALL_COLOR = (0, 0, 255)
PADDLE_COLOR = (0, 255, 0)

# Load Font
font = pygame.font.SysFont("Times New Roman", 40, bold=True)

# Ball and Paddle
ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, 20, 20)
ball_speed = [4, -4]
paddle = pygame.Rect(WIDTH // 2 - 60, HEIGHT - 30, 120, 10)
prev_hand_x = None
lives = 3  # Introduce lives system

# Start Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam. Please ensure your webcam is connected and accessible.")

running = True
game_started = False  # Game starts only after a key press
game_over = False

def reset_game():
    """Reset the game state."""
    global ball, ball_speed, lives, game_started, game_over
    ball.x, ball.y = WIDTH // 2, HEIGHT // 2
    ball_speed = [4, -4]
    lives = 3
    game_started = False
    game_over = False

while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Detect Hands
    result = hands.process(rgb_frame)
    hand_landmarks = result.multi_hand_landmarks

    # Convert OpenCV frame to Pygame Surface
    frame = cv2.resize(frame, (WIDTH, HEIGHT))  # Resize the frame to match the screen size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure the frame is in RGB format
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))  # Transpose the frame for Pygame

    if not game_over:
        if hand_landmarks:
            for hand_landmarks_obj in hand_landmarks:
                hand_x_positions = [lm.x for lm in hand_landmarks_obj.landmark]
                avg_x = int(sum(hand_x_positions) / len(hand_x_positions) * WIDTH)

                if prev_hand_x is not None:
                    paddle_movement = avg_x - prev_hand_x
                    paddle.centerx += paddle_movement  # Move in the same direction as hand
                    paddle.centerx = max(60, min(WIDTH - 60, paddle.centerx))
                prev_hand_x = avg_x

                mp_draw.draw_landmarks(frame, hand_landmarks_obj, mp_hands.HAND_CONNECTIONS)

        if game_started:
            # Update ball position
            ball.x += ball_speed[0]
            ball.y += ball_speed[1]

            # Ball collision with walls
            if ball.left <= 0 or ball.right >= WIDTH:
                ball_speed[0] *= -1
            if ball.top <= 0:
                ball_speed[1] *= -1

            # Ball collision with paddle
            if ball.colliderect(paddle) and ball_speed[1] > 0:
                ball_speed[1] *= -1
                ball_speed[0] += random.choice([-1, 1])  # Add small random variation

            # Ball falls below the paddle
            if ball.bottom >= HEIGHT:
                lives -= 1
                if lives <= 0:
                    game_over = True
                    game_started = False
                else:
                    ball.x, ball.y = WIDTH // 2, HEIGHT // 2
                    ball_speed[1] = -4  # Reset ball movement upwards

    # Render the game
    screen.blit(frame_surface, (0, 0))

    if game_started and not game_over:
        pygame.draw.rect(screen, PADDLE_COLOR, paddle)
        pygame.draw.ellipse(screen, BALL_COLOR, ball)
        lives_surface = font.render(f"Lives: {lives}", True, WHITE)
        screen.blit(lives_surface, (10, 10))
    elif not game_started and not game_over:
        # Display "Press SPACE to Start" message
        start_surface = font.render("Press SPACE to Start", True, WHITE)
        start_rect = start_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(start_surface, start_rect)

    if game_over:
        text_surface = font.render("GAME OVER - Press R to Restart", True, RED)
        glow_surface = font.render("GAME OVER - Press R to Restart", True, GOLD)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        glow_rect = glow_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            screen.blit(glow_surface, (glow_rect.x + dx, glow_rect.y + dy))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()
    clock.tick(60)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_started and not game_over:
                game_started = True  # Start the game when SPACE is pressed
            elif event.key == pygame.K_r and game_over:
                reset_game()  # Restart the game if 'R' is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
