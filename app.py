"""
app.py -- Thumb Games Hub
===========================
A menu to pick from different games, all controlled by your thumb
angle via webcam (0.0 = thumb down, 1.0 = thumb up, everything
in between mapped continuously).

Currently available:
    - Pong (vs AI or 2P) -- your paddle height follows your thumb value

HOW TO RUN
-----------
1. Make sure these are all in the same folder:
     - thumb_regressor.pth   (your trained model)
     - thumb_control.py      (webcam/model controller module)
     - app.py                (this file)

2. Install pygame if you don't have it (run in Thonny's shell):
     pip install pygame

3. Run this file (F5 in Thonny). A window opens with the menu.

4. Click "Pong" to play:
     - Choose "1 Player (vs AI)" or "2 Player"
     - 1 Player: also choose a difficulty (Easy / Medium / Hard) by
       dragging the slider or clicking a position, then press Start
       - Thumb straight up   -> your paddle at the TOP
       - Thumb straight down -> your paddle at the BOTTOM
     - 2 Player: starts immediately
       - P1's LEFT hand controls the left (blue) paddle
       - P2's RIGHT hand controls the right (red) paddle
       - (matches the blue=left / red=right convention used in the
         two-hand webcam preview)
     - Press ESC anytime to back out a step (or all the way to menu
       from the game itself)

5. On the menu, press Q or close the window to quit entirely.

OPTIONAL -- USING THE REAL "OSWALD" FONT
-------------------------------------------
The UI uses a font called Oswald for a cleaner look. If it's not
installed on your system, everything still works fine with a plain
fallback font. To get the real Oswald look:
  1. Download it for free from https://fonts.google.com/specimen/Oswald
  2. Unzip it and copy "Oswald-Bold.ttf" and "Oswald-Regular.ttf"
     into this same folder (next to app.py)
  3. Re-run the app -- it'll automatically pick up the local files.
No download? No problem -- the app falls back to a clean system font
automatically, nothing breaks.

ADDING MORE GAMES LATER
-------------------------
Every game is a class with handle_event(), update(dt), and draw()
methods (see BaseGame / PongGame below). To add a new game:
   1. Write a new class following the same pattern as PongGame,
      using self.controller.get_value() for thumb input.
   2. Add it to the GAMES dictionary near the bottom of this file,
      including an icon emoji/character and fallback symbol:
        GAMES = {
            "Pong": {"class": PongGame, "icon": "\U0001F3AE", "fallback_icon": "\u25B6"},
        }
   It'll automatically show up as a button in the menu.
"""

import os
import sys
import math
import random
import threading
import pygame

# NOTE: thumb_control is NOT imported here at module level on purpose --
# it pulls in PyTorch/MediaPipe/OpenCV, which are slow to import. Importing
# it up front would block window creation for several seconds before
# anything even appears on screen. Instead it's imported lazily in a
# background thread inside main() -- see _load_controller() below -- so the
# window and menu show up immediately with a small loading screen instead.

# ============================================================
# WINDOW / COLOR CONSTANTS
# ============================================================

WINDOW_W, WINDOW_H = 900, 600
FPS = 60

WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
GRAY = (140, 145, 160)
DARK_GRAY = (60, 64, 78)
GREEN = (70, 210, 130)
YELLOW = (230, 200, 60)
RED = (225, 80, 80)
BLUE = (80, 150, 235)
ACCENT = (90, 200, 255)

TOP_BG = (14, 20, 48)      # dark blue
BOTTOM_BG = (6, 6, 10)     # almost black

# ============================================================
# FONT LOADING (tries local Oswald .ttf files, falls back gracefully)
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_FILES = {
    "bold": ["Oswald-Bold.ttf", "Oswald-SemiBold.ttf"],
    "regular": ["Oswald-Regular.ttf", "Oswald-Medium.ttf", "Oswald-Light.ttf"],
}


def load_app_font(weight, size):
    """Load Oswald from a local .ttf if present, else fall back to an
    installed 'Oswald' system font, else a plain default sans font."""
    candidates = FONT_FILES.get(weight, FONT_FILES["regular"])
    for fname in candidates:
        path = os.path.join(_THIS_DIR, fname)
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
    try:
        f = pygame.font.SysFont("oswald", size, bold=(weight == "bold"))
        if f is not None:
            return f
    except Exception:
        pass
    return pygame.font.SysFont("arial", size, bold=(weight == "bold"))


# ============================================================
# EMOJI / ICON HELPERS
# ============================================================

_EMOJI_FONT_CANDIDATES = ["Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji"]


def _surface_has_content(surf, threshold=40):
    try:
        alpha = pygame.surfarray.array_alpha(surf)
        return alpha.sum() > threshold
    except Exception:
        return False


def render_emoji(char, size):
    """Try to render a real color emoji glyph. Returns a Surface or
    None if no emoji-capable font produced visible output."""
    for name in _EMOJI_FONT_CANDIDATES:
        try:
            font = pygame.font.SysFont(name, size)
            surf = font.render(char, True, (255, 255, 255))
            if _surface_has_content(surf):
                return surf
        except Exception:
            continue
    return None


def render_icon(emoji_char, fallback_char, size, font, color):
    """Try a real emoji first; if unavailable, render the fallback
    character (a normal Unicode symbol, e.g. an arrow) using the
    given UI font/color so it always matches the button style."""
    surf = render_emoji(emoji_char, size)
    if surf is not None:
        return surf
    return font.render(fallback_char, True, color)


def make_gradient(width, height, top_color, bottom_color):
    surf = pygame.Surface((width, height))
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (width, y))
    return surf


def make_thumb_icon_surface(size=64):
    """Try a real thumbs-up emoji for the window icon; fall back to a
    simple hand-drawn thumbs-up shape if no emoji font is available."""
    surf = render_emoji("\U0001F44D", size)  # 👍
    if surf is not None:
        return pygame.transform.smoothscale(surf, (size, size))

    # Fallback: draw a simple stylized thumbs-up icon
    icon = pygame.Surface((size, size), pygame.SRCALPHA)
    skin = (240, 190, 120)
    outline = (90, 60, 20)
    # fist
    fist_rect = pygame.Rect(size * 0.30, size * 0.45, size * 0.45, size * 0.42)
    pygame.draw.rect(icon, skin, fist_rect, border_radius=int(size * 0.12))
    pygame.draw.rect(icon, outline, fist_rect, width=2, border_radius=int(size * 0.12))
    # thumb
    thumb_rect = pygame.Rect(size * 0.32, size * 0.08, size * 0.22, size * 0.42)
    pygame.draw.rect(icon, skin, thumb_rect, border_radius=int(size * 0.10))
    pygame.draw.rect(icon, outline, thumb_rect, width=2, border_radius=int(size * 0.10))
    return icon


# ============================================================
# REUSABLE ANIMATED BUTTON
# ============================================================

class Button:
    """Rounded button with a subtle shadow, hover-grow animation, and
    a darker fill while being clicked."""

    def __init__(self, rect, text, font, icon_emoji=None, icon_fallback=None,
                 base_color=GREEN, text_color=BLACK):
        self.base_rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.base_color = base_color
        self.text_color = text_color
        self.scale = 1.0
        self.hovered = False
        self.pressed = False

        self.icon_surface = None
        if icon_emoji or icon_fallback:
            self.icon_surface = render_icon(
                icon_emoji or "", icon_fallback or "", int(self.base_rect.h * 0.55),
                font, text_color,
            )

    def update(self, dt, mouse_pos, mouse_down):
        self.hovered = self.base_rect.collidepoint(mouse_pos)
        self.pressed = self.hovered and mouse_down
        target_scale = 1.06 if self.hovered else 1.0
        self.scale += (target_scale - self.scale) * min(1.0, dt * 14)

    def is_clicked(self, pos):
        return self.base_rect.collidepoint(pos)

    def _scaled_rect(self):
        w = int(self.base_rect.w * self.scale)
        h = int(self.base_rect.h * self.scale)
        r = pygame.Rect(0, 0, w, h)
        r.center = self.base_rect.center
        return r

    def draw(self, screen):
        rect = self._scaled_rect()
        radius = int(rect.h * 0.28)

        # subtle drop shadow
        shadow = pygame.Surface((rect.w + 16, rect.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 90), (8, 12, rect.w, rect.h), border_radius=radius)
        screen.blit(shadow, (rect.x - 8, rect.y - 6))

        if self.pressed:
            color = tuple(max(0, c - 55) for c in self.base_color)
        elif self.hovered:
            color = tuple(min(255, c + 20) for c in self.base_color)
        else:
            color = self.base_color

        pygame.draw.rect(screen, color, rect, border_radius=radius)
        pygame.draw.rect(screen, (255, 255, 255, 60), rect, width=2, border_radius=radius)

        text_surf = self.font.render(self.text, True, self.text_color)
        content_w = text_surf.get_width()
        icon_gap = 0
        if self.icon_surface is not None:
            icon_gap = self.icon_surface.get_width() + 12
            content_w += icon_gap

        start_x = rect.centerx - content_w // 2
        if self.icon_surface is not None:
            icon_y = rect.centery - self.icon_surface.get_height() // 2
            screen.blit(self.icon_surface, (start_x, icon_y))
            start_x += icon_gap

        text_y = rect.centery - text_surf.get_height() // 2
        screen.blit(text_surf, (start_x, text_y))


# ============================================================
# GAME INTERFACE -- every game implements these methods
# ============================================================

class BaseGame:
    name = "Base Game"
    SUPPORTS_DIFFICULTY = False    # set True on subclasses with an AI difficulty setting
    SUPPORTS_TWO_PLAYER = False    # set True on subclasses with a 2-player mode

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        self.screen = screen
        self.controller = controller
        self.difficulty = difficulty  # "Easy" / "Medium" / "Hard" / None
        self.mode = mode              # "AI" or "2P"

    def handle_event(self, event):
        """Handle a single pygame event (keypress, click, etc)."""
        pass

    def update(self, dt):
        """Advance game state by dt seconds. Read thumb value via
        self.controller.get_value()."""
        pass

    def draw(self):
        """Draw the current frame to self.screen."""
        pass


# ============================================================
# PONG
# ============================================================

class PongGame(BaseGame):
    name = "Pong"
    SUPPORTS_DIFFICULTY = True
    SUPPORTS_TWO_PLAYER = True

    PADDLE_W, PADDLE_H = 15, 100
    BALL_SIZE = 16
    LEFT_X = 40
    RIGHT_X = WINDOW_W - 40 - PADDLE_W

    BALL_SPEED_INCREASE = 25  # ball speeds up slightly on every paddle hit
    WIN_SCORE = 7
    TWO_PLAYER_BALL_SPEED = 340  # ball speed used in 2-player mode (no AI difficulty to derive it from)

    DIFFICULTY_SETTINGS = {
        "Easy":   {"ai_speed": 160, "ai_slack": 50, "ball_speed": 280},
        "Medium": {"ai_speed": 260, "ai_slack": 20, "ball_speed": 320},
        "Hard":   {"ai_speed": 420, "ai_slack": 5,  "ball_speed": 380},
    }

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)

        if self.mode == "AI":
            settings = self.DIFFICULTY_SETTINGS.get(self.difficulty, self.DIFFICULTY_SETTINGS["Medium"])
            self.AI_SPEED = settings["ai_speed"]
            self.AI_REACTION_SLACK = settings["ai_slack"]
            self.BALL_SPEED_START = settings["ball_speed"]
        else:  # "2P"
            self.BALL_SPEED_START = self.TWO_PLAYER_BALL_SPEED

        self.font_big = load_app_font("bold", 60)
        self.font_med = load_app_font("bold", 32)
        self.font_small = load_app_font("regular", 20)
        self.reset_match()

    def reset_match(self):
        self.left_score = 0
        self.right_score = 0
        self.left_y = WINDOW_H / 2 - self.PADDLE_H / 2
        self.right_y = WINDOW_H / 2 - self.PADDLE_H / 2
        self.game_over = False
        self.reset_ball(direction=random.choice([-1, 1]))

    def reset_ball(self, direction=1):
        self.ball_x = WINDOW_W / 2 - self.BALL_SIZE / 2
        self.ball_y = WINDOW_H / 2 - self.BALL_SIZE / 2
        self.ball_vx = self.BALL_SPEED_START * direction
        self.ball_vy = self.BALL_SPEED_START * random.uniform(-0.5, 0.5)

    def handle_event(self, event):
        if self.game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_match()

    def update(self, dt):
        if self.game_over:
            return

        if self.mode == "AI":
            # ---- left paddle: human, driven by whichever hand is seen ----
            value = max(0.0, min(1.0, self.controller.get_value()))
            self.left_y = (1.0 - value) * (WINDOW_H - self.PADDLE_H)

            # ---- right paddle: AI chases the ball with a speed cap + dead zone ----
            ai_center = self.right_y + self.PADDLE_H / 2
            diff = self.ball_y + self.BALL_SIZE / 2 - ai_center
            if abs(diff) > self.AI_REACTION_SLACK:
                move = self.AI_SPEED * dt
                if diff > 0:
                    self.right_y += min(move, diff)
                else:
                    self.right_y -= min(move, -diff)
            self.right_y = max(0, min(WINDOW_H - self.PADDLE_H, self.right_y))

        else:  # "2P" -- left hand controls left paddle, right hand controls right paddle
            left_value = max(0.0, min(1.0, self.controller.get_left_value()))
            right_value = max(0.0, min(1.0, self.controller.get_right_value()))
            self.left_y = (1.0 - left_value) * (WINDOW_H - self.PADDLE_H)
            self.right_y = (1.0 - right_value) * (WINDOW_H - self.PADDLE_H)

        # ---- ball physics (same for both modes) ----
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        if self.ball_y <= 0 or self.ball_y >= WINDOW_H - self.BALL_SIZE:
            self.ball_vy *= -1
            self.ball_y = max(0, min(WINDOW_H - self.BALL_SIZE, self.ball_y))

        left_rect = pygame.Rect(self.LEFT_X, self.left_y, self.PADDLE_W, self.PADDLE_H)
        right_rect = pygame.Rect(self.RIGHT_X, self.right_y, self.PADDLE_W, self.PADDLE_H)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        if ball_rect.colliderect(left_rect) and self.ball_vx < 0:
            self._bounce_off_paddle(left_rect, going_right=True)
        elif ball_rect.colliderect(right_rect) and self.ball_vx > 0:
            self._bounce_off_paddle(right_rect, going_right=False)

        if self.ball_x < -self.BALL_SIZE:
            self.right_score += 1
            self._check_game_over()
            if not self.game_over:
                self.reset_ball(direction=1)
        elif self.ball_x > WINDOW_W:
            self.left_score += 1
            self._check_game_over()
            if not self.game_over:
                self.reset_ball(direction=-1)

    def _bounce_off_paddle(self, paddle_rect, going_right):
        hit_pos = (self.ball_y + self.BALL_SIZE / 2 - paddle_rect.centery) / (self.PADDLE_H / 2)
        hit_pos = max(-1.0, min(1.0, hit_pos))

        speed = abs(self.ball_vx) + self.BALL_SPEED_INCREASE
        self.ball_vx = speed if going_right else -speed
        self.ball_vy = speed * 0.75 * hit_pos

        if going_right:
            self.ball_x = paddle_rect.right + 1
        else:
            self.ball_x = paddle_rect.left - self.BALL_SIZE - 1

    def _check_game_over(self):
        if self.left_score >= self.WIN_SCORE or self.right_score >= self.WIN_SCORE:
            self.game_over = True

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))

        for y in range(0, WINDOW_H, 30):
            pygame.draw.rect(self.screen, DARK_GRAY, (WINDOW_W // 2 - 2, y, 4, 15))

        pygame.draw.rect(self.screen, BLUE, (self.LEFT_X, self.left_y, self.PADDLE_W, self.PADDLE_H), border_radius=4)
        pygame.draw.rect(self.screen, RED, (self.RIGHT_X, self.right_y, self.PADDLE_W, self.PADDLE_H), border_radius=4)
        pygame.draw.ellipse(self.screen, WHITE, (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        score_text = self.font_big.render(f"{self.left_score}   {self.right_score}", True, WHITE)
        self.screen.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 20))

        if self.mode == "AI":
            left_label, right_label = "YOU", "AI"
        else:
            left_label, right_label = "P1", "P2"

        label_left = self.font_small.render(left_label, True, BLUE)
        label_right = self.font_small.render(right_label, True, RED)
        self.screen.blit(label_left, (self.LEFT_X, 10))
        self.screen.blit(label_right, (self.RIGHT_X, 10))

        if self.mode == "AI":
            mode_text = self.font_small.render(f"Difficulty: {self.difficulty or 'Medium'}", True, GRAY)
        else:
            mode_text = self.font_small.render("2 Player Mode", True, GRAY)
        self.screen.blit(mode_text, (WINDOW_W // 2 - mode_text.get_width() // 2, WINDOW_H - 55))

        if self.mode == "AI":
            hint = "Thumb UP = paddle top, DOWN = paddle bottom | ESC = menu"
        else:
            hint = "P1: left hand controls left paddle | P2: right hand controls right paddle | ESC = menu"
        hint_text = self.font_small.render(hint, True, GRAY)
        self.screen.blit(hint_text, (WINDOW_W // 2 - hint_text.get_width() // 2, WINDOW_H - 30))

        if self.mode == "AI":
            if not self.controller.get_hand_detected():
                warn = self.font_small.render("No hand detected -- show your thumb to the webcam", True, RED)
                self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, 60))
        else:
            if not self.controller.get_left_detected():
                warn = self.font_small.render("P1: no hand detected (left side)", True, BLUE)
                self.screen.blit(warn, (self.LEFT_X, 60))
            if not self.controller.get_right_detected():
                warn = self.font_small.render("P2: no hand detected (right side)", True, RED)
                self.screen.blit(warn, (WINDOW_W - self.RIGHT_X - warn.get_width(), 60))

        if self.game_over:
            if self.mode == "AI":
                winner = "YOU WIN!" if self.left_score > self.right_score else "AI WINS!"
            else:
                winner = "P1 WINS!" if self.left_score > self.right_score else "P2 WINS!"
            color = GREEN if self.left_score > self.right_score else RED
            win_text = self.font_big.render(winner, True, color)
            self.screen.blit(win_text, (WINDOW_W // 2 - win_text.get_width() // 2, WINDOW_H // 2 - 60))

            retry_text = self.font_med.render("Press R to play again, ESC for menu", True, WHITE)
            self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 10))


# ============================================================
# BREAKOUT
# ============================================================

class BreakoutGame(BaseGame):
    name = "Breakout"
    SUPPORTS_DIFFICULTY = False
    SUPPORTS_TWO_PLAYER = False

    PADDLE_W, PADDLE_H = 110, 16
    PADDLE_Y = WINDOW_H - 40
    BALL_SIZE = 14
    BALL_SPEED_START = 320
    BALL_SPEED_INCREASE = 12   # ball speeds up slightly on every brick hit
    BALL_SPEED_MAX = 560

    LIVES_START = 3

    BRICK_ROWS = 5
    BRICK_COLS = 10
    BRICK_W, BRICK_H = 80, 24
    BRICK_GAP = 6
    BRICK_TOP = 80
    BRICK_LEFT = (WINDOW_W - (BRICK_COLS * BRICK_W + (BRICK_COLS - 1) * BRICK_GAP)) // 2

    # top row = hardest to reach = most points, matching the classic look
    ROW_COLORS = [RED, YELLOW, GREEN, BLUE, ACCENT]
    ROW_POINTS = [7, 5, 3, 2, 1]

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)

        self.font_big = load_app_font("bold", 46)
        self.font_med = load_app_font("bold", 26)
        self.font_small = load_app_font("regular", 20)

        self.reset_game()

    def reset_game(self):
        self.score = 0
        self.lives = self.LIVES_START
        self.game_over = False
        self.won = False
        self.paddle_x = WINDOW_W / 2 - self.PADDLE_W / 2
        self.time_since_brick = 0.0

        self.bricks = []
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                x = self.BRICK_LEFT + col * (self.BRICK_W + self.BRICK_GAP)
                y = self.BRICK_TOP + row * (self.BRICK_H + self.BRICK_GAP)
                self.bricks.append({
                    "rect": pygame.Rect(x, y, self.BRICK_W, self.BRICK_H),
                    "color": self.ROW_COLORS[row],
                    "points": self.ROW_POINTS[row],
                })

        self.reset_ball()

    def reset_ball(self):
        self.ball_x = WINDOW_W / 2 - self.BALL_SIZE / 2
        self.ball_y = self.PADDLE_Y - self.BALL_SIZE - 4
        self.ball_vx = self.BALL_SPEED_START * random.choice([-1, 1]) * random.uniform(0.4, 0.7)
        self.ball_vy = -self.BALL_SPEED_START * random.uniform(0.8, 1.0)

    def handle_event(self, event):
        if self.game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_game()

    def _jitter_bounce_angle(self, max_degrees=4.0):
        """Nudge the ball's direction by a small random angle. Pure
        axis-aligned reflection can otherwise settle into a perfectly
        periodic loop (e.g. bouncing forever in a gap between two
        surviving bricks) -- a tiny bit of noise on every bounce keeps
        that from ever becoming a permanent trap."""
        speed = math.hypot(self.ball_vx, self.ball_vy)
        angle = math.atan2(self.ball_vy, self.ball_vx) + math.radians(random.uniform(-max_degrees, max_degrees))
        self.ball_vx = math.cos(angle) * speed
        self.ball_vy = math.sin(angle) * speed

    def update(self, dt):
        # thumb controls the paddle's horizontal position: thumb UP -> right,
        # thumb DOWN -> left, every value in between mapped continuously
        value = max(0.0, min(1.0, self.controller.get_value()))
        self.paddle_x = value * (WINDOW_W - self.PADDLE_W)

        if self.game_over:
            return

        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        # side/top walls
        if self.ball_x <= 0:
            self.ball_x = 0
            self.ball_vx *= -1
            self._jitter_bounce_angle()
        elif self.ball_x >= WINDOW_W - self.BALL_SIZE:
            self.ball_x = WINDOW_W - self.BALL_SIZE
            self.ball_vx *= -1
            self._jitter_bounce_angle()
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1
            self._jitter_bounce_angle()

        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        # paddle bounce (only while the ball is moving down into it)
        paddle_rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_W, self.PADDLE_H)
        if self.ball_vy > 0 and ball_rect.colliderect(paddle_rect):
            hit_pos = (self.ball_x + self.BALL_SIZE / 2 - paddle_rect.centerx) / (self.PADDLE_W / 2)
            hit_pos = max(-1.0, min(1.0, hit_pos))
            speed = min(self.BALL_SPEED_MAX, math.hypot(self.ball_vx, self.ball_vy))
            self.ball_vy = -abs(speed) * 0.85
            self.ball_vx = speed * 0.75 * hit_pos
            self.ball_y = paddle_rect.top - self.BALL_SIZE - 1

        # brick collisions -- only resolve one brick per frame
        hit_brick = False
        for brick in self.bricks:
            if ball_rect.colliderect(brick["rect"]):
                self.bricks.remove(brick)
                self.score += brick["points"]
                hit_brick = True

                # bounce off whichever side was hit
                overlap_x = min(ball_rect.right, brick["rect"].right) - max(ball_rect.left, brick["rect"].left)
                overlap_y = min(ball_rect.bottom, brick["rect"].bottom) - max(ball_rect.top, brick["rect"].top)
                if overlap_x < overlap_y:
                    self.ball_vx *= -1
                else:
                    self.ball_vy *= -1

                speed = min(self.BALL_SPEED_MAX, math.hypot(self.ball_vx, self.ball_vy) + self.BALL_SPEED_INCREASE)
                angle = math.atan2(self.ball_vy, self.ball_vx)
                self.ball_vx = math.cos(angle) * speed
                self.ball_vy = math.sin(angle) * speed
                self._jitter_bounce_angle()
                break

        # safety net: with only a few scattered bricks left, a purely
        # physics-driven ball can occasionally rattle around for ages without
        # ever crossing paths with them. If nothing's been hit in a while,
        # bend its direction partway toward the nearest survivor -- a
        # directed nudge (not just more randomness) so it reliably
        # converges instead of hoping for lucky bounces.
        if self.bricks:
            self.time_since_brick = 0.0 if hit_brick else self.time_since_brick + dt
            if self.time_since_brick > 6.0:
                nearest = min(self.bricks, key=lambda b: (b["rect"].centerx - self.ball_x) ** 2
                              + (b["rect"].centery - self.ball_y) ** 2)
                target_angle = math.atan2(nearest["rect"].centery - self.ball_y,
                                           nearest["rect"].centerx - self.ball_x)
                cur_angle = math.atan2(self.ball_vy, self.ball_vx)
                angle_gap = (target_angle - cur_angle + math.pi) % (2 * math.pi) - math.pi
                speed = math.hypot(self.ball_vx, self.ball_vy)
                new_angle = cur_angle + angle_gap * 0.85
                self.ball_vx = math.cos(new_angle) * speed
                self.ball_vy = math.sin(new_angle) * speed
                self.time_since_brick = 0.0
        else:
            self.game_over = True
            self.won = True

        # ball fell below the paddle
        if self.ball_y > WINDOW_H:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.won = False
            else:
                self.reset_ball()

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))

        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"], border_radius=4)
            pygame.draw.rect(self.screen, (255, 255, 255, 60), brick["rect"], width=1, border_radius=4)

        paddle_rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_W, self.PADDLE_H)
        pygame.draw.rect(self.screen, BLUE, paddle_rect, border_radius=4)
        pygame.draw.ellipse(self.screen, WHITE, (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        score_text = self.font_big.render(f"{self.score}", True, WHITE)
        self.screen.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 16))

        lives_text = self.font_small.render(f"Lives: {self.lives}", True, GRAY)
        self.screen.blit(lives_text, (WINDOW_W - lives_text.get_width() - 20, 20))

        if not self.controller.get_hand_detected():
            warn = self.font_small.render("No hand detected -- show your thumb to the webcam", True, RED)
            self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, 60))

        hint = self.font_small.render(
            "Thumb UP = paddle right, DOWN = paddle left | ESC = menu", True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 20))

        if self.game_over:
            text = "YOU WIN!" if self.won else "GAME OVER"
            color = GREEN if self.won else RED
            win_text = self.font_big.render(text, True, color)
            self.screen.blit(win_text, (WINDOW_W // 2 - win_text.get_width() // 2, WINDOW_H // 2 - 70))

            score_line = self.font_med.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_line, (WINDOW_W // 2 - score_line.get_width() // 2, WINDOW_H // 2 - 10))

            retry_text = self.font_small.render("Press R to play again, ESC for menu", True, WHITE)
            self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 30))


# ============================================================
# BALANCE BEAM
# ============================================================

class BalanceBeamGame(BaseGame):
    """Keep a ball balanced on a tiltable beam for as long as possible.
    Thumb UP tilts the beam right, DOWN tilts it left, 0.5 is level --
    every value in between maps to a proportional tilt angle.

    The beam starts each run already tilted at a random angle (fading back
    to purely thumb-controlled over the first couple of seconds), so just
    holding the thumb neutral and coasting isn't an option -- you have to
    react from the very first frame. Random gusts of "wind" keep nudging
    the ball afterward, growing stronger the longer you survive, and small
    obstacles fall from the top of the screen -- get bonked by one and the
    ball takes a knock. Difficulty (Easy/Medium/Hard) scales how strong the
    wind/starting tilt/obstacles are."""

    name = "Balance Beam"
    SUPPORTS_DIFFICULTY = True
    SUPPORTS_TWO_PLAYER = False

    BEAM_HALF_LENGTH = 300
    BEAM_THICKNESS = 8
    PIVOT = (WINDOW_W / 2, WINDOW_H / 2 + 30)
    MAX_TILT = math.radians(30)
    BALL_RADIUS = 13
    FALL_MARGIN = 6  # ball must go slightly past the end before it actually falls

    GRAVITY = 480.0    # px/sec^2 pulling the ball downhill at full tilt
    DAMPING = 0.24     # fraction of velocity bled off per second (rolling friction)

    GRACE_PERIOD = 1.5       # seconds before wind gusts start
    WIND_CHANGE_INTERVAL = (1.2, 2.4)  # random seconds between picking a new wind target
    WIND_SMOOTHING = 3.0     # how fast current wind eases toward its target

    BIAS_DECAY_TIME = 2.0    # seconds for the random starting tilt to fade to pure thumb control

    OBSTACLE_RADIUS = 11
    OBSTACLE_SPAWN_MARGIN = 30  # keep obstacles spawning within reach of the beam's ends
    KNOCK_STRENGTH = 105.0      # velocity kick (px/sec) from getting hit by an obstacle
    HIT_FLASH_TIME = 0.35

    # Only the hazards scale with difficulty -- control feel (MAX_TILT,
    # GRAVITY, DAMPING) stays the same across all three so your sense of
    # "how the beam responds" doesn't change, only how much it's tested.
    DIFFICULTY_SETTINGS = {
        "Easy": {
            "wind_base": 32.0, "wind_growth": 2.6, "wind_max": 200.0,
            "start_bias_frac": 0.42, "obstacle_interval": (2.0, 2.9), "obstacle_speed": 230.0,
        },
        "Medium": {
            "wind_base": 55.0, "wind_growth": 5.5, "wind_max": 320.0,
            "start_bias_frac": 0.62, "obstacle_interval": (1.4, 2.1), "obstacle_speed": 300.0,
        },
        "Hard": {
            "wind_base": 75.0, "wind_growth": 8.0, "wind_max": 420.0,
            "start_bias_frac": 0.82, "obstacle_interval": (0.9, 1.4), "obstacle_speed": 370.0,
        },
    }

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)
        self.difficulty = difficulty or "Medium"

        self.font_big = load_app_font("bold", 46)
        self.font_med = load_app_font("bold", 26)
        self.font_small = load_app_font("regular", 20)

        self.best_time = 0.0
        self.reset_game()

    def reset_game(self):
        settings = self.DIFFICULTY_SETTINGS.get(self.difficulty, self.DIFFICULTY_SETTINGS["Medium"])
        self.wind_base = settings["wind_base"]
        self.wind_growth = settings["wind_growth"]
        self.wind_max = settings["wind_max"]
        self.obstacle_interval = settings["obstacle_interval"]
        self.obstacle_speed = settings["obstacle_speed"]

        self.s = 0.0
        self.vel = 0.0
        self.tilt = 0.0
        self.time_survived = 0.0
        self.game_over = False
        self.wind_current = 0.0
        self.wind_target = 0.0
        self._wind_timer = random.uniform(*self.WIND_CHANGE_INTERVAL)

        # beam starts already tilted a random amount (recoverable -- always
        # well within what the player can counter with full thumb input)
        # and fades to purely thumb-controlled over BIAS_DECAY_TIME
        self.start_tilt_bias = random.uniform(-1.0, 1.0) * settings["start_bias_frac"] * self.MAX_TILT

        self.obstacles = []  # each: {"x", "y"}
        self._obstacle_timer = random.uniform(*self.obstacle_interval)
        self.hit_flash = 0.0

    def handle_event(self, event):
        if self.game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.reset_game()

    def update(self, dt):
        # thumb controls the beam's tilt: thumb UP -> tilts right,
        # DOWN -> tilts left, every value in between mapped continuously
        value = max(0.0, min(1.0, self.controller.get_value()))
        input_tilt = (value - 0.5) * 2 * self.MAX_TILT

        if self.game_over:
            self.tilt = input_tilt
            return

        self.time_survived += dt
        self.hit_flash = max(0.0, self.hit_flash - dt)

        bias_decay = max(0.0, 1.0 - self.time_survived / self.BIAS_DECAY_TIME)
        self.tilt = input_tilt + self.start_tilt_bias * bias_decay

        # wind: smoothly drifts toward a new random target every couple of
        # seconds, and its range widens the longer you've survived
        self._wind_timer -= dt
        if self._wind_timer <= 0:
            wind_range = min(self.wind_max, self.wind_base + self.time_survived * self.wind_growth)
            self.wind_target = random.uniform(-wind_range, wind_range)
            self._wind_timer = random.uniform(*self.WIND_CHANGE_INTERVAL)
        self.wind_current += (self.wind_target - self.wind_current) * min(1.0, self.WIND_SMOOTHING * dt)

        wind_effect = 0.0 if self.time_survived < self.GRACE_PERIOD else self.wind_current

        accel = self.GRAVITY * math.sin(self.tilt) + wind_effect
        self.vel += accel * dt
        self.vel *= max(0.0, 1.0 - self.DAMPING * dt)
        self.s += self.vel * dt

        self._update_obstacles(dt)

        if abs(self.s) > self.BEAM_HALF_LENGTH + self.FALL_MARGIN:
            self.game_over = True
            self.best_time = max(self.best_time, self.time_survived)

    def _update_obstacles(self, dt):
        self._obstacle_timer -= dt
        if self._obstacle_timer <= 0:
            cx, _ = self.PIVOT
            lo = cx - self.BEAM_HALF_LENGTH + self.OBSTACLE_SPAWN_MARGIN
            hi = cx + self.BEAM_HALF_LENGTH - self.OBSTACLE_SPAWN_MARGIN
            self.obstacles.append({"x": random.uniform(lo, hi), "y": -20.0})
            self._obstacle_timer = random.uniform(*self.obstacle_interval)

        ball_x, ball_y = self._ball_pos()
        survivors = []
        for ob in self.obstacles:
            ob["y"] += self.obstacle_speed * dt
            if ob["y"] > WINDOW_H + 20:
                continue  # fell past the bottom, missed
            dist = math.hypot(ob["x"] - ball_x, ob["y"] - ball_y)
            if dist < self.BALL_RADIUS + self.OBSTACLE_RADIUS:
                push_dir = 1.0 if ball_x >= ob["x"] else -1.0
                self.vel += push_dir * self.KNOCK_STRENGTH
                self.hit_flash = self.HIT_FLASH_TIME
                continue  # consumed on impact
            survivors.append(ob)
        self.obstacles = survivors

    def _beam_endpoints(self):
        cx, cy = self.PIVOT
        cos_t, sin_t = math.cos(self.tilt), math.sin(self.tilt)
        left = (cx - self.BEAM_HALF_LENGTH * cos_t, cy - self.BEAM_HALF_LENGTH * sin_t)
        right = (cx + self.BEAM_HALF_LENGTH * cos_t, cy + self.BEAM_HALF_LENGTH * sin_t)
        return left, right

    def _ball_pos(self):
        cx, cy = self.PIVOT
        cos_t, sin_t = math.cos(self.tilt), math.sin(self.tilt)
        along_x, along_y = cx + self.s * cos_t, cy + self.s * sin_t
        up_x, up_y = sin_t, -cos_t  # perpendicular to the beam, pointing "up" off its surface
        offset = self.BEAM_THICKNESS / 2 + self.BALL_RADIUS
        return along_x + up_x * offset, along_y + up_y * offset

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))

        cx, cy = self.PIVOT
        # fulcrum stand under the pivot
        pygame.draw.polygon(self.screen, DARK_GRAY, [(cx - 26, cy + 70), (cx + 26, cy + 70), (cx, cy + 10)])
        pygame.draw.line(self.screen, GRAY, (cx - 60, cy + 70), (cx + 60, cy + 70), 4)

        left, right = self._beam_endpoints()
        pygame.draw.line(self.screen, BLUE, left, right, self.BEAM_THICKNESS)
        pygame.draw.circle(self.screen, BLUE, (int(left[0]), int(left[1])), self.BEAM_THICKNESS // 2)
        pygame.draw.circle(self.screen, BLUE, (int(right[0]), int(right[1])), self.BEAM_THICKNESS // 2)
        pygame.draw.circle(self.screen, DARK_GRAY, (int(cx), int(cy)), 7)

        for ob in self.obstacles:
            pygame.draw.circle(self.screen, YELLOW, (int(ob["x"]), int(ob["y"])), self.OBSTACLE_RADIUS)
            pygame.draw.circle(self.screen, DARK_GRAY, (int(ob["x"]), int(ob["y"])), self.OBSTACLE_RADIUS, 2)

        ball_x, ball_y = self._ball_pos()
        near_edge = abs(self.s) > self.BEAM_HALF_LENGTH * 0.8
        ball_color = RED if near_edge else WHITE
        pygame.draw.circle(self.screen, ball_color, (int(ball_x), int(ball_y)), self.BALL_RADIUS)
        pygame.draw.circle(self.screen, DARK_GRAY, (int(ball_x), int(ball_y)), self.BALL_RADIUS, 2)
        if self.hit_flash > 0:
            pygame.draw.circle(self.screen, YELLOW, (int(ball_x), int(ball_y)), self.BALL_RADIUS + 5, 2)

        time_text = self.font_big.render(f"{self.time_survived:0.1f}s", True, WHITE)
        self.screen.blit(time_text, (WINDOW_W // 2 - time_text.get_width() // 2, 16))

        best_text = self.font_small.render(f"Best: {self.best_time:0.1f}s", True, GRAY)
        self.screen.blit(best_text, (WINDOW_W - best_text.get_width() - 20, 20))

        if not self.controller.get_hand_detected():
            warn = self.font_small.render("No hand detected -- show your thumb to the webcam", True, RED)
            self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, 60))

        hint = self.font_small.render(
            f"Thumb UP = tilt right, DOWN = tilt left | Difficulty: {self.difficulty} | ESC = menu",
            True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 20))

        if self.game_over:
            over_text = self.font_big.render("GAME OVER", True, RED)
            self.screen.blit(over_text, (WINDOW_W // 2 - over_text.get_width() // 2, WINDOW_H // 2 - 70))

            time_line = self.font_med.render(f"You lasted {self.time_survived:0.1f}s", True, WHITE)
            self.screen.blit(time_line, (WINDOW_W // 2 - time_line.get_width() // 2, WINDOW_H // 2 - 10))

            retry_text = self.font_small.render("Press R to play again, ESC for menu", True, WHITE)
            self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 30))


# ============================================================
# AIR HOCKEY
# ============================================================

class AirHockeyGame(BaseGame):
    """Air hockey vs an AI opponent. Your mallet slides along the bottom
    edge, the AI's along the top edge -- thumb UP moves you right, DOWN
    moves you left, every value in between mapped continuously (same
    control scheme as Breakout's paddle). First to WIN_SCORE goals wins."""

    name = "Air Hockey"
    SUPPORTS_DIFFICULTY = True
    SUPPORTS_TWO_PLAYER = False

    RINK_MARGIN = 40
    RINK_LEFT = RINK_MARGIN
    RINK_RIGHT = WINDOW_W - RINK_MARGIN
    RINK_TOP = RINK_MARGIN
    RINK_BOTTOM = WINDOW_H - RINK_MARGIN

    GOAL_HALF_WIDTH = 90  # goal mouth spans the center +/- this, in the top/bottom walls

    MALLET_RADIUS = 26
    PUCK_RADIUS = 10
    PLAYER_Y = RINK_BOTTOM - 40
    AI_Y = RINK_TOP + 40

    PUCK_SPEED_INCREASE = 18   # puck speeds up slightly on every mallet hit
    PUCK_SPEED_MAX = 560
    WIN_SCORE = 5   # goals are hard-fought (small mallet, narrow goal) -- 7 made matches drag, especially on Hard

    DIFFICULTY_SETTINGS = {
        "Easy":   {"ai_speed": 220, "ai_slack": 60, "puck_speed": 260},
        "Medium": {"ai_speed": 320, "ai_slack": 30, "puck_speed": 300},
        "Hard":   {"ai_speed": 460, "ai_slack": 10, "puck_speed": 340},
    }

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)
        self.difficulty = difficulty or "Medium"

        settings = self.DIFFICULTY_SETTINGS.get(self.difficulty, self.DIFFICULTY_SETTINGS["Medium"])
        self.AI_SPEED = settings["ai_speed"]
        self.AI_REACTION_SLACK = settings["ai_slack"]
        self.PUCK_SPEED_START = settings["puck_speed"]

        self.font_big = load_app_font("bold", 46)
        self.font_med = load_app_font("bold", 26)
        self.font_small = load_app_font("regular", 18)

        self._build_ice_texture()
        self.reset_match()

    def _build_ice_texture(self):
        """Pre-render the rink's icy surface once (gradient + shine streaks
        + frost cracks) so draw() just blits it instead of recomputing it
        every frame. Colors stay in the same blue/cyan family as ACCENT and
        BLUE elsewhere in the app -- just brighter, like a lit rink in a
        dark arena, so it fits the rest of the theme instead of looking
        like a random bright rectangle dropped on top of it."""
        w = int(self.RINK_RIGHT - self.RINK_LEFT)
        h = int(self.RINK_BOTTOM - self.RINK_TOP)
        ice_top = (150, 205, 230)
        ice_bottom = (95, 165, 200)

        surf = pygame.Surface((w, h))
        for y in range(h):
            t = y / max(1, h - 1)
            color = tuple(int(ice_top[i] + (ice_bottom[i] - ice_top[i]) * t) for i in range(3))
            pygame.draw.line(surf, color, (0, y), (w, y))

        # soft diagonal shine streaks, like light glinting off the ice
        shine = pygame.Surface((w, h), pygame.SRCALPHA)
        for x in range(-h, w, 90):
            pygame.draw.line(shine, (255, 255, 255, 22), (x, 0), (x + h, h), 34)
        surf.blit(shine, (0, 0))

        # a handful of faint frost cracks -- fixed seed so the pattern is
        # stable across frames instead of flickering
        rng = random.Random(2024)
        for _ in range(7):
            x, y = rng.randint(0, w), rng.randint(0, h)
            points = [(x, y)]
            for _ in range(rng.randint(2, 4)):
                x = max(0, min(w, x + rng.randint(-45, 45)))
                y = max(0, min(h, y + rng.randint(-45, 45)))
                points.append((x, y))
            pygame.draw.lines(surf, (180, 220, 240), False, points, 1)

        self._ice_surface = surf

    def reset_match(self):
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.player_mallet_x = WINDOW_W / 2
        self.ai_mallet_x = WINDOW_W / 2
        self.reset_puck()

    def reset_puck(self):
        self.puck_x = WINDOW_W / 2
        self.puck_y = WINDOW_H / 2
        # bias the serve toward a decent vertical component so rallies
        # reliably head toward a goal instead of drifting sideways forever
        horiz_frac = random.uniform(-0.6, 0.6)
        vert_frac = math.sqrt(max(0.01, 1.0 - horiz_frac ** 2)) * random.choice([-1, 1])
        self.puck_vx = self.PUCK_SPEED_START * horiz_frac
        self.puck_vy = self.PUCK_SPEED_START * vert_frac

    def handle_event(self, event):
        if self.game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.reset_match()

    def update(self, dt):
        # thumb controls the mallet's horizontal position: thumb UP -> right,
        # DOWN -> left, every value in between mapped continuously
        value = max(0.0, min(1.0, self.controller.get_value()))
        left_bound = self.RINK_LEFT + self.MALLET_RADIUS
        right_bound = self.RINK_RIGHT - self.MALLET_RADIUS
        self.player_mallet_x = left_bound + value * (right_bound - left_bound)

        if self.game_over:
            return

        # ---- AI mallet: chases the puck's x (speed cap + dead zone) once the
        # puck is in its own half where it's an actual threat; otherwise it
        # drifts back to defend the center of its goal, same as a sensible
        # human player naturally does. Without this it stays glued to the
        # puck's x even while the puck is all the way down at the other end,
        # leaving its own goal wide open the instant the puck comes back. ----
        target_x = self.puck_x if self.puck_y < WINDOW_H / 2 else WINDOW_W / 2
        diff = target_x - self.ai_mallet_x
        if abs(diff) > self.AI_REACTION_SLACK:
            move = self.AI_SPEED * dt
            if diff > 0:
                self.ai_mallet_x += min(move, diff)
            else:
                self.ai_mallet_x -= min(move, -diff)
        self.ai_mallet_x = max(left_bound, min(right_bound, self.ai_mallet_x))

        # ---- puck physics ----
        self.puck_x += self.puck_vx * dt
        self.puck_y += self.puck_vy * dt

        # mallets are resolved BEFORE the wall/goal checks so the wall check
        # always gets the final say for the frame. Doing it the other way
        # around (walls first) let a mallet parked near a corner reflect the
        # puck straight back into the wall it had just bounced off of --
        # wall bounce and mallet bounce fighting each other forever with the
        # puck stuck oscillating in the corner and never reaching the goal.
        self._resolve_mallet_hit(self.player_mallet_x, self.PLAYER_Y)
        self._resolve_mallet_hit(self.ai_mallet_x, self.AI_Y)

        if self.puck_x - self.PUCK_RADIUS < self.RINK_LEFT:
            self.puck_x = self.RINK_LEFT + self.PUCK_RADIUS
            if self.puck_vx < 0:
                self.puck_vx *= -1
                self._jitter_puck_angle()
        elif self.puck_x + self.PUCK_RADIUS > self.RINK_RIGHT:
            self.puck_x = self.RINK_RIGHT - self.PUCK_RADIUS
            if self.puck_vx > 0:
                self.puck_vx *= -1
                self._jitter_puck_angle()

        in_goal_mouth = abs(self.puck_x - WINDOW_W / 2) < self.GOAL_HALF_WIDTH

        if self.puck_y - self.PUCK_RADIUS < self.RINK_TOP:
            if in_goal_mouth:
                self._score(scorer="player")
            else:
                self.puck_y = self.RINK_TOP + self.PUCK_RADIUS
                if self.puck_vy < 0:
                    self.puck_vy *= -1
                    self._jitter_puck_angle()
        elif self.puck_y + self.PUCK_RADIUS > self.RINK_BOTTOM:
            if in_goal_mouth:
                self._score(scorer="ai")
            else:
                self.puck_y = self.RINK_BOTTOM - self.PUCK_RADIUS
                if self.puck_vy > 0:
                    self.puck_vy *= -1
                    self._jitter_puck_angle()

    def _score(self, scorer):
        if scorer == "player":
            self.player_score += 1
        else:
            self.ai_score += 1
        if self.player_score >= self.WIN_SCORE or self.ai_score >= self.WIN_SCORE:
            self.game_over = True
        else:
            self.reset_puck()

    def _resolve_mallet_hit(self, mallet_x, mallet_y):
        dx = self.puck_x - mallet_x
        dy = self.puck_y - mallet_y
        dist = math.hypot(dx, dy)
        if dist >= self.MALLET_RADIUS + self.PUCK_RADIUS:
            return
        if dist < 1e-6:
            dx, dy, dist = 0.0, -1.0, 1.0
        nx, ny = dx / dist, dy / dist  # collision normal, mallet center -> puck center

        # a mallet parked near a corner can otherwise bounce the puck
        # straight into the adjacent wall, trapping it there indefinitely
        # (wall bounce and mallet bounce fighting forever). Sign-flipping a
        # unit component doesn't change the vector's length, so this stays
        # a valid unit normal -- never let the bounce point into a wall
        # anywhere near where the puck currently is. The threshold is
        # MALLET_RADIUS-wide (not just "touching") because the puck can be
        # hit anywhere within the mallet's reach, and the mallet itself can
        # be parked with its edge nearly against the wall.
        wall_buffer = self.MALLET_RADIUS
        if self.puck_x - self.PUCK_RADIUS <= self.RINK_LEFT + wall_buffer and nx < 0:
            nx = -nx
        elif self.puck_x + self.PUCK_RADIUS >= self.RINK_RIGHT - wall_buffer and nx > 0:
            nx = -nx
        if self.puck_y - self.PUCK_RADIUS <= self.RINK_TOP + wall_buffer and ny < 0:
            ny = -ny
        elif self.puck_y + self.PUCK_RADIUS >= self.RINK_BOTTOM - wall_buffer and ny > 0:
            ny = -ny

        speed = min(self.PUCK_SPEED_MAX, math.hypot(self.puck_vx, self.puck_vy) + self.PUCK_SPEED_INCREASE)
        self.puck_vx = nx * speed
        self.puck_vy = ny * speed

        # push the puck fully outside the mallet so it can't keep re-colliding next frame
        overlap = (self.MALLET_RADIUS + self.PUCK_RADIUS) - dist
        self.puck_x += nx * overlap
        self.puck_y += ny * overlap
        self._jitter_puck_angle()

    def _jitter_puck_angle(self, max_degrees=3.0):
        """Nudge the puck's direction by a small random angle on every
        bounce. Since mallets only move horizontally, a puck that happens
        to collide dead-center (same x as the mallet) reflects perfectly
        vertically -- and can stay perfectly vertical forever, endlessly
        bouncing between the walls/mallets outside the goal mouth without
        ever scoring. A tiny bit of noise breaks that exact alignment."""
        speed = math.hypot(self.puck_vx, self.puck_vy)
        angle = math.atan2(self.puck_vy, self.puck_vx) + math.radians(random.uniform(-max_degrees, max_degrees))
        self.puck_vx = math.cos(angle) * speed
        self.puck_vy = math.sin(angle) * speed

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))
        self.screen.blit(self._ice_surface, (self.RINK_LEFT, self.RINK_TOP))

        rink_rect = pygame.Rect(self.RINK_LEFT, self.RINK_TOP,
                                 self.RINK_RIGHT - self.RINK_LEFT, self.RINK_BOTTOM - self.RINK_TOP)
        pygame.draw.rect(self.screen, DARK_GRAY, rink_rect, width=3, border_radius=6)

        # center line + circle
        pygame.draw.line(self.screen, DARK_GRAY, (self.RINK_LEFT, WINDOW_H // 2), (self.RINK_RIGHT, WINDOW_H // 2), 2)
        pygame.draw.circle(self.screen, DARK_GRAY, (WINDOW_W // 2, WINDOW_H // 2), 55, 2)

        # goal mouths, highlighted
        goal_left = WINDOW_W // 2 - self.GOAL_HALF_WIDTH
        goal_right = WINDOW_W // 2 + self.GOAL_HALF_WIDTH
        pygame.draw.line(self.screen, RED, (goal_left, self.RINK_TOP), (goal_right, self.RINK_TOP), 4)
        pygame.draw.line(self.screen, BLUE, (goal_left, self.RINK_BOTTOM), (goal_right, self.RINK_BOTTOM), 4)

        pygame.draw.circle(self.screen, RED, (int(self.ai_mallet_x), int(self.AI_Y)), self.MALLET_RADIUS)
        pygame.draw.circle(self.screen, WHITE, (int(self.ai_mallet_x), int(self.AI_Y)), self.MALLET_RADIUS, 2)
        pygame.draw.circle(self.screen, BLUE, (int(self.player_mallet_x), int(self.PLAYER_Y)), self.MALLET_RADIUS)
        pygame.draw.circle(self.screen, WHITE, (int(self.player_mallet_x), int(self.PLAYER_Y)), self.MALLET_RADIUS, 2)

        pygame.draw.circle(self.screen, WHITE, (int(self.puck_x), int(self.puck_y)), self.PUCK_RADIUS)
        pygame.draw.circle(self.screen, DARK_GRAY, (int(self.puck_x), int(self.puck_y)), self.PUCK_RADIUS, 2)

        score_text = self.font_med.render(f"{self.player_score}   {self.ai_score}", True, WHITE)
        self.screen.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 8))

        label_ai = self.font_small.render("AI", True, RED)
        self.screen.blit(label_ai, (WINDOW_W // 2 - 100, self.RINK_TOP + 6))
        label_you = self.font_small.render("YOU", True, BLUE)
        self.screen.blit(label_you, (WINDOW_W // 2 - 100, self.RINK_BOTTOM - 26))

        if not self.controller.get_hand_detected():
            warn = self.font_small.render("No hand detected -- show your thumb to the webcam", True, RED)
            self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, WINDOW_H // 2 - 90))

        hint = self.font_small.render(
            f"Thumb UP = mallet right, DOWN = mallet left | Difficulty: {self.difficulty} | ESC = menu",
            True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 20))

        if self.game_over:
            winner = "YOU WIN!" if self.player_score > self.ai_score else "AI WINS!"
            color = GREEN if self.player_score > self.ai_score else RED
            win_text = self.font_big.render(winner, True, color)
            self.screen.blit(win_text, (WINDOW_W // 2 - win_text.get_width() // 2, WINDOW_H // 2 - 60))

            retry_text = self.font_med.render("Press R to play again, ESC for menu", True, WHITE)
            self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 10))


# ============================================================
# THUMB GOLF
# ============================================================

class ThumbGolfGame(BaseGame):
    """One-hole-at-a-time golf, played across a 3-hole round.

    Two-hand control: RIGHT thumb sets shot power live (1.0 = full power,
    every value in between scales proportionally), LEFT thumb fires the
    shot -- raise it into a thumbs-up and the ball launches using whatever
    power the right thumb reads at that instant. The shot always aims
    toward the hole, so overshooting just means the next shot needs less
    power, no separate aiming control needed.

    Modes: 1 Player (solo), 2 Player (take turns on the same hole before
    moving to the next), or vs AI (an intentionally weak AI opponent).
    Each difficulty has its own set of 3 themed holes (meadow/desert/snow)
    with a different hazard each -- a sand trap, a water hazard, and a
    crosswind -- and each difficulty's holes start further from the tee.
    """

    name = "Thumb Golf"
    SUPPORTS_DIFFICULTY = False  # handled internally: needs a 3-way mode
    SUPPORTS_TWO_PLAYER = False  # choice, not the generic 1P-vs-AI/2P split

    COURSE_MARGIN = 40
    COURSE_LEFT = COURSE_MARGIN
    COURSE_RIGHT = WINDOW_W - COURSE_MARGIN
    COURSE_TOP = 90
    COURSE_BOTTOM = WINDOW_H - COURSE_MARGIN
    GROUND_Y = 470

    TEE_X = COURSE_LEFT + 40
    BALL_RADIUS = 9
    HOLE_RADIUS = 16

    LAUNCH_ANGLE = math.radians(38)
    GRAVITY = 900.0
    MIN_SHOT_SPEED = 80.0
    MAX_SHOT_SPEED = 640.0
    LANDING_DAMPING = 0.82   # fraction of horizontal speed kept after the bounce-down on landing
    ROLL_FRICTION = 380.0    # px/sec^2 deceleration while rolling on the fairway
    SAND_ROLL_FRICTION = 1400.0  # much stronger -- a ball landing in sand stops fast
    ROLL_STOP_SPEED = 4.0

    SHOOT_THRESHOLD = 0.75      # LEFT thumb value above this counts as "thumbs up" (fire)
    CAPTURE_MAX_SPEED = 240.0   # a ball rolling faster than this skips over the cup instead of dropping in

    DISTANCE_UNIT_SCALE = 1.0 / 3.0  # purely cosmetic: display px as golf-y "yards"

    HOLES_PER_ROUND = 3
    MAX_STROKES_PER_HOLE = 8   # mercy rule so a bad run (or a deliberately bad AI) can't drag on forever
    TURN_TRANSITION_TIME = 1.8  # seconds the "holed in X!" message shows before the next turn/hole

    AI_THINK_TIME = 1.0       # seconds of "lining up" pause before the AI swings
    AI_POWER_NOISE = 0.22     # how wildly the "bad" AI misjudges its power -- this IS the bad AI
    AI_DISTANCE_REFERENCE = 665.0  # ~= max distance a full-power shot travels, for the AI's rough estimate

    # tuned so a well-calibrated (not maxed-out) swing reaches each hole:
    # roughly 62% power for Easy, 80% for Medium, 96% for Hard
    DIFFICULTY_SETTINGS = {
        "Easy":   {"hole_distance": 260, "par": 2},
        "Medium": {"hole_distance": 430, "par": 3},
        "Hard":   {"hole_distance": 610, "par": 4},
    }

    # one entry per hole in the round -- theme + hazard, shared across all
    # three difficulties (only hole_distance/par scale with difficulty)
    HOLE_DEFS = [
        {"theme": "meadow", "obstacle": "sand", "obstacle_frac": 0.55, "obstacle_width_frac": 0.14},
        {"theme": "desert", "obstacle": "water", "obstacle_frac": 0.60, "obstacle_width_frac": 0.11},
        {"theme": "snow", "obstacle": "wind", "wind_accel": 70.0},
    ]

    THEME_COLORS = {
        "meadow": {
            "sky_top": (150, 205, 240), "sky_bottom": (205, 228, 246),
            "ground_top": (80, 190, 120), "ground_bottom": (45, 140, 85),
            "horizon": (60, 170, 100),
        },
        "desert": {
            "sky_top": (245, 210, 165), "sky_bottom": (255, 235, 205),
            "ground_top": (225, 195, 140), "ground_bottom": (185, 150, 95),
            "horizon": (195, 160, 100),
        },
        "snow": {
            "sky_top": (205, 222, 238), "sky_bottom": (232, 238, 246),
            "ground_top": (235, 240, 248), "ground_bottom": (205, 218, 232),
            "horizon": (190, 205, 222),
        },
    }

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)

        self.font_big = load_app_font("bold", 40)
        self.font_med = load_app_font("bold", 26)
        self.font_small = load_app_font("regular", 18)
        self.font_tiny = load_app_font("regular", 15)
        self._thumbsup_icon = render_icon("\U0001F44D", "UP", 22, self.font_tiny, WHITE)  # 👍, cached once

        self._course_textures = {theme: self._build_course_texture(theme) for theme in self.THEME_COLORS}

        self.match_mode = "1P"
        self.difficulty = "Medium"
        self.phase = "MODE_SELECT"
        self._build_mode_buttons()

    # ---------------------------------------------------------------
    # course art
    # ---------------------------------------------------------------

    def _build_course_texture(self, theme):
        """Pre-render one theme's sky+fairway once (cached, blitted each
        frame) instead of recomputing it every frame."""
        colors = self.THEME_COLORS[theme]
        w = int(self.COURSE_RIGHT - self.COURSE_LEFT)
        h = int(self.COURSE_BOTTOM - self.COURSE_TOP)
        sky_h = int(self.GROUND_Y - self.COURSE_TOP)
        grass_h = h - sky_h

        surf = pygame.Surface((w, h))
        for y in range(sky_h):
            t = y / max(1, sky_h - 1)
            color = tuple(int(colors["sky_top"][i] + (colors["sky_bottom"][i] - colors["sky_top"][i]) * t)
                          for i in range(3))
            pygame.draw.line(surf, color, (0, y), (w, y))
        for y in range(grass_h):
            t = y / max(1, grass_h - 1)
            color = tuple(int(colors["ground_top"][i] + (colors["ground_bottom"][i] - colors["ground_top"][i]) * t)
                          for i in range(3))
            pygame.draw.line(surf, color, (0, sky_h + y), (w, sky_h + y))
        pygame.draw.line(surf, colors["horizon"], (0, sky_h), (w, sky_h), 3)

        rng = random.Random(hash(theme) & 0xFFFF)  # stable per-theme seed, no per-frame flicker
        if theme == "meadow":
            clouds = pygame.Surface((w, h), pygame.SRCALPHA)
            for _ in range(5):
                cx, cy = rng.randint(30, w - 30), rng.randint(20, max(21, sky_h // 2))
                for dx, dy, r in ((0, 0, 18), (16, 4, 14), (-14, 3, 13), (6, -8, 11)):
                    pygame.draw.circle(clouds, (255, 255, 255, 90), (cx + dx, cy + dy), r)
            surf.blit(clouds, (0, 0))
        elif theme == "desert":
            cactus_color = (70, 140, 80)
            for _ in range(4):
                cx = rng.randint(20, w - 20)
                cy = sky_h + rng.randint(6, max(7, grass_h // 3))
                pygame.draw.rect(surf, cactus_color, (cx - 4, cy - 26, 8, 30), border_radius=4)
                pygame.draw.rect(surf, cactus_color, (cx - 14, cy - 14, 8, 16), border_radius=4)
                pygame.draw.rect(surf, cactus_color, (cx + 6, cy - 18, 8, 20), border_radius=4)
        elif theme == "snow":
            tree_color = (40, 95, 65)
            for _ in range(4):
                tx = rng.randint(20, w - 20)
                ty = sky_h + rng.randint(4, max(5, grass_h // 3))
                pygame.draw.polygon(surf, tree_color, [(tx, ty - 34), (tx - 16, ty - 6), (tx + 16, ty - 6)])
                pygame.draw.polygon(surf, tree_color, [(tx, ty - 24), (tx - 13, ty + 2), (tx + 13, ty + 2)])
                pygame.draw.rect(surf, (90, 60, 40), (tx - 3, ty, 6, 8))

        return surf

    # ---------------------------------------------------------------
    # mode / difficulty select (self-contained -- no shared ModeSelect/
    # DifficultySelect screens, since Golf needs a 3-way mode choice and
    # BOTH 2P and vs-AI also need a difficulty/hole-set choice, unlike
    # Pong's simpler AI-only-picks-difficulty flow)
    # ---------------------------------------------------------------

    def _build_mode_buttons(self):
        btn_w, btn_h, gap = 320, 64, 20
        total_h = 3 * btn_h + 2 * gap
        start_y = WINDOW_H // 2 - total_h // 2 + 20
        specs = [
            ("1P", "1 Player", "\U0001F464", "1"),
            ("2P", "2 Player", "\U0001F465", "2"),
            ("AI", "vs AI", "\U0001F916", "A"),
        ]
        self.mode_buttons = []
        for i, (mode_id, label, icon, fallback) in enumerate(specs):
            rect = pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y + i * (btn_h + gap), btn_w, btn_h)
            btn = Button(rect, label, self.font_med, icon_emoji=icon, icon_fallback=fallback,
                         base_color=GREEN, text_color=BLACK)
            self.mode_buttons.append((btn, mode_id))

    def _build_difficulty_buttons(self):
        btn_w, btn_h, gap = 320, 64, 20
        total_h = 3 * btn_h + 2 * gap
        start_y = WINDOW_H // 2 - total_h // 2 + 20
        specs = [("Easy", GREEN), ("Medium", YELLOW), ("Hard", RED)]
        self.difficulty_buttons = []
        for i, (label, color) in enumerate(specs):
            rect = pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y + i * (btn_h + gap), btn_w, btn_h)
            btn = Button(rect, label, self.font_med, base_color=color, text_color=BLACK)
            self.difficulty_buttons.append((btn, label))
        self.back_button = Button(pygame.Rect(30, 30, 110, 46), "Back", self.font_small,
                                   icon_fallback="←", base_color=DARK_GRAY, text_color=WHITE)

    def _draw_mode_select(self):
        self.screen.blit(BACKGROUND, (0, 0))
        title = self.font_big.render("Thumb Golf", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 90))
        subtitle = self.font_small.render("Choose how you want to play", True, GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 150))
        for btn, _ in self.mode_buttons:
            btn.draw(self.screen)

    def _draw_difficulty_select(self):
        self.screen.blit(BACKGROUND, (0, 0))
        title = self.font_big.render("Choose Difficulty", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 90))
        subtitle = self.font_small.render(
            "3 holes each -- harder difficulties start further from the tee", True, GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 150))
        for btn, _ in self.difficulty_buttons:
            btn.draw(self.screen)
        self.back_button.draw(self.screen)

    # ---------------------------------------------------------------
    # match / turn / hole setup
    # ---------------------------------------------------------------

    def start_round(self):
        if self.match_mode == "1P":
            self.players = ["YOU"]
        elif self.match_mode == "2P":
            self.players = ["P1", "P2"]
        else:  # "AI"
            self.players = ["YOU", "AI"]
        self.scores = {p: [] for p in self.players}
        self.hole_idx = 0
        self.turn_idx = 0
        self._setup_hole()
        self.phase = "PLAYING"

    def _setup_hole(self):
        hole_def = self.HOLE_DEFS[self.hole_idx]
        settings = self.DIFFICULTY_SETTINGS[self.difficulty]
        self.par = settings["par"]
        hole_distance = settings["hole_distance"]
        self.hole_x = self.TEE_X + hole_distance

        self.theme = hole_def["theme"]
        self.obstacle_type = hole_def.get("obstacle")
        self.wind_accel = 0.0
        self.obstacle_left = self.obstacle_right = None
        if self.obstacle_type in ("sand", "water"):
            center = self.TEE_X + hole_distance * hole_def["obstacle_frac"]
            half_w = hole_distance * hole_def["obstacle_width_frac"] / 2
            self.obstacle_left = center - half_w
            self.obstacle_right = center + half_w
        elif self.obstacle_type == "wind":
            self.wind_accel = hole_def["wind_accel"]

        self._course_surface = self._course_textures[self.theme]
        self.turn_idx = 0
        self._start_player_turn()

    def _start_player_turn(self):
        self.ball_x = float(self.TEE_X)
        self.ball_y = float(self.GROUND_Y)
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_state = "ready"
        self.strokes = 0
        self.holed = False
        self.last_shot_power = None
        self.power_live = 0.0
        self._prev_left_value = None
        self.ai_timer = None

    def _current_player(self):
        return self.players[self.turn_idx]

    def _current_player_is_ai(self):
        return self._current_player() == "AI"

    def handle_event(self, event):
        if self.phase == "MODE_SELECT":
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, mode_id in self.mode_buttons:
                    if btn.is_clicked(event.pos):
                        self.match_mode = mode_id
                        self._build_difficulty_buttons()
                        self.phase = "DIFFICULTY_SELECT"
                        return
        elif self.phase == "DIFFICULTY_SELECT":
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, label in self.difficulty_buttons:
                    if btn.is_clicked(event.pos):
                        self.difficulty = label
                        self.start_round()
                        return
                if self.back_button.is_clicked(event.pos):
                    self._build_mode_buttons()
                    self.phase = "MODE_SELECT"
                    return
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._build_mode_buttons()
                self.phase = "MODE_SELECT"

    # ---------------------------------------------------------------
    # update
    # ---------------------------------------------------------------

    def update(self, dt):
        if self.phase in ("MODE_SELECT", "DIFFICULTY_SELECT"):
            mouse_pos = pygame.mouse.get_pos()
            mouse_down = pygame.mouse.get_pressed()[0]
            buttons = self.mode_buttons if self.phase == "MODE_SELECT" else self.difficulty_buttons
            for btn, _ in buttons:
                btn.update(dt, mouse_pos, mouse_down)
            if self.phase == "DIFFICULTY_SELECT":
                self.back_button.update(dt, mouse_pos, mouse_down)
        elif self.phase == "PLAYING":
            self._update_playing(dt)
        elif self.phase == "TURN_COMPLETE":
            self.turn_transition_timer -= dt
            if self.turn_transition_timer <= 0:
                self._advance_after_turn()
        # ROUND_COMPLETE: nothing to update, just waiting for R

    def _update_playing(self, dt):
        if self.ball_state == "ready" and not self.holed:
            if self._current_player_is_ai():
                if self.ai_timer is None:
                    self.ai_timer = self.AI_THINK_TIME
                else:
                    self.ai_timer -= dt
                    if self.ai_timer <= 0:
                        self._take_ai_shot()
                        self.ai_timer = None
            else:
                right_value = max(0.0, min(1.0, self.controller.get_right_value()))
                left_value = max(0.0, min(1.0, self.controller.get_left_value()))
                self.power_live = right_value

                if self._prev_left_value is None:
                    self._prev_left_value = left_value
                else:
                    if self._prev_left_value <= self.SHOOT_THRESHOLD < left_value:
                        self._take_shot(right_value)
                    self._prev_left_value = left_value
        elif self.ball_state != "ready":
            self._update_flight(dt)
            near_ground = abs(self.ball_y - self.GROUND_Y) < 4
            slow_enough = abs(self.ball_vx) < self.CAPTURE_MAX_SPEED
            if near_ground and slow_enough and abs(self.ball_x - self.hole_x) < self.HOLE_RADIUS:
                self.holed = True
                self.ball_state = "ready"
                self.ball_x = self.hole_x
                self.ball_y = self.GROUND_Y

        if self.ball_state == "ready" and (self.holed or self.strokes >= self.MAX_STROKES_PER_HOLE):
            self._end_turn()

    def _take_shot(self, power):
        self.last_shot_power = power
        speed = self.MIN_SHOT_SPEED + power * (self.MAX_SHOT_SPEED - self.MIN_SHOT_SPEED)
        direction = 1.0 if self.hole_x >= self.ball_x else -1.0
        self.ball_vx = direction * speed * math.cos(self.LAUNCH_ANGLE)
        self.ball_vy = -speed * math.sin(self.LAUNCH_ANGLE)
        self.ball_state = "flying"
        self.strokes += 1

    def _take_ai_shot(self):
        # deliberately bad: a rough distance-based estimate plus a lot of noise
        remaining = abs(self.hole_x - self.ball_x)
        target_power = max(0.0, min(1.0, math.sqrt(remaining / self.AI_DISTANCE_REFERENCE)))
        noisy_power = target_power + random.uniform(-self.AI_POWER_NOISE, self.AI_POWER_NOISE)
        noisy_power = max(0.05, min(1.0, noisy_power))
        self._take_shot(noisy_power)

    def _update_flight(self, dt):
        if self.ball_state == "flying":
            self.ball_vy += self.GRAVITY * dt
            if self.wind_accel:
                self.ball_vx += self.wind_accel * dt
            self.ball_x += self.ball_vx * dt
            self.ball_y += self.ball_vy * dt
            if self.ball_y >= self.GROUND_Y:
                self.ball_y = self.GROUND_Y
                self.ball_vx *= self.LANDING_DAMPING
                self.ball_vy = 0.0
                self.ball_state = "rolling"
                self._check_water_landing()
        elif self.ball_state == "rolling":
            friction = self.ROLL_FRICTION
            if (self.obstacle_type == "sand" and self.obstacle_left is not None
                    and self.obstacle_left <= self.ball_x <= self.obstacle_right):
                friction = self.SAND_ROLL_FRICTION
            self.ball_x += self.ball_vx * dt
            if self.ball_vx > 0:
                self.ball_vx = max(0.0, self.ball_vx - friction * dt)
            else:
                self.ball_vx = min(0.0, self.ball_vx + friction * dt)
            if abs(self.ball_vx) < self.ROLL_STOP_SPEED:
                self.ball_vx = 0.0
                self.ball_state = "ready"
            self._check_water_landing()

        self.ball_x = max(self.COURSE_LEFT + self.BALL_RADIUS,
                           min(self.COURSE_RIGHT - self.BALL_RADIUS, self.ball_x))

    def _check_water_landing(self):
        if (self.obstacle_type != "water" or self.obstacle_left is None
                or not (self.obstacle_left <= self.ball_x <= self.obstacle_right)):
            return
        # splash -- one-stroke penalty, dropped just short of the hazard
        self.strokes += 1
        drop_x = self.obstacle_left - 20 if self.ball_vx >= 0 else self.obstacle_right + 20
        self.ball_x = max(self.COURSE_LEFT + self.BALL_RADIUS,
                           min(self.COURSE_RIGHT - self.BALL_RADIUS, drop_x))
        self.ball_y = self.GROUND_Y
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_state = "ready"

    def _end_turn(self):
        player = self._current_player()
        self.scores[player].append(self.strokes)
        self.turn_result_holed = self.holed
        self.turn_result_strokes = self.strokes
        self.turn_result_player = player
        self.phase = "TURN_COMPLETE"
        self.turn_transition_timer = self.TURN_TRANSITION_TIME

    def _advance_after_turn(self):
        self.turn_idx += 1
        if self.turn_idx >= len(self.players):
            self.turn_idx = 0
            self.hole_idx += 1
            if self.hole_idx >= self.HOLES_PER_ROUND:
                self.phase = "ROUND_COMPLETE"
                return
            self._setup_hole()
        else:
            self._start_player_turn()
        self.phase = "PLAYING"

    # ---------------------------------------------------------------
    # draw
    # ---------------------------------------------------------------

    def draw(self):
        if self.phase == "MODE_SELECT":
            self._draw_mode_select()
        elif self.phase == "DIFFICULTY_SELECT":
            self._draw_difficulty_select()
        elif self.phase in ("PLAYING", "TURN_COMPLETE"):
            self._draw_playing()
            if self.phase == "TURN_COMPLETE":
                self._draw_turn_complete_overlay()
        elif self.phase == "ROUND_COMPLETE":
            self._draw_playing()
            self._draw_round_complete_overlay()

    def _draw_playing(self):
        self.screen.blit(BACKGROUND, (0, 0))
        self.screen.blit(self._course_surface, (self.COURSE_LEFT, self.COURSE_TOP))
        self._draw_obstacle()

        pygame.draw.circle(self.screen, WHITE, (int(self.TEE_X), int(self.GROUND_Y)), 4)

        flag_top = self.GROUND_Y - 55
        pygame.draw.ellipse(self.screen, (25, 25, 30), (self.hole_x - 10, self.GROUND_Y - 4, 20, 8))
        pygame.draw.line(self.screen, GRAY, (self.hole_x, self.GROUND_Y), (self.hole_x, flag_top), 3)
        pygame.draw.polygon(self.screen, RED, [(self.hole_x, flag_top),
                                                (self.hole_x + 22, flag_top + 9),
                                                (self.hole_x, flag_top + 18)])

        pygame.draw.circle(self.screen, WHITE, (int(self.ball_x), int(self.ball_y)), self.BALL_RADIUS)
        pygame.draw.circle(self.screen, DARK_GRAY, (int(self.ball_x), int(self.ball_y)), self.BALL_RADIUS, 2)

        if self._current_player_is_ai():
            self._draw_ai_turn_indicator()
        else:
            self._draw_power_meter()
            self._draw_shoot_indicator()

        self._draw_scoreboard()
        self._draw_hud()

    def _draw_obstacle(self):
        if self.obstacle_left is None:
            return
        rect = (self.obstacle_left, self.GROUND_Y - 6, self.obstacle_right - self.obstacle_left, 16)
        if self.obstacle_type == "sand":
            pygame.draw.ellipse(self.screen, (210, 185, 130), rect)
            pygame.draw.ellipse(self.screen, (180, 155, 100), rect, 2)
        elif self.obstacle_type == "water":
            pygame.draw.ellipse(self.screen, (80, 160, 200), rect)
            pygame.draw.ellipse(self.screen, (50, 120, 160), rect, 2)

    def _draw_power_meter(self):
        x, y, w, h = 24, 110, 26, 220
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, w, h), border_radius=8)

        fill_h = int(h * self.power_live)
        if fill_h > 0:
            fill_color = GREEN if self.power_live < 0.55 else (YELLOW if self.power_live < 0.82 else RED)
            pygame.draw.rect(self.screen, fill_color, (x, y + h - fill_h, w, fill_h), border_radius=8)
        pygame.draw.rect(self.screen, GRAY, (x, y, w, h), width=2, border_radius=8)

        if self.last_shot_power is not None:
            marker_y = y + h - int(h * self.last_shot_power)
            pygame.draw.line(self.screen, WHITE, (x - 5, marker_y), (x + w + 5, marker_y), 2)

        label = self.font_tiny.render("POWER", True, GRAY)
        self.screen.blit(label, (x + w // 2 - label.get_width() // 2, y - 20))
        hand_label = self.font_tiny.render("RIGHT", True, GRAY)
        self.screen.blit(hand_label, (x + w // 2 - hand_label.get_width() // 2, y + h + 6))

    def _draw_shoot_indicator(self):
        x, y, r = 37, 372, 20
        ready = self._prev_left_value is not None and self._prev_left_value > self.SHOOT_THRESHOLD
        color = GREEN if ready else DARK_GRAY
        pygame.draw.circle(self.screen, color, (x, y), r)
        pygame.draw.circle(self.screen, GRAY, (x, y), r, 2)

        icon = self._thumbsup_icon
        self.screen.blit(icon, (x - icon.get_width() // 2, y - icon.get_height() // 2))

        label = self.font_tiny.render("SHOOT", True, GRAY)
        self.screen.blit(label, (x - label.get_width() // 2, y + r + 6))
        hand_label = self.font_tiny.render("LEFT", True, GRAY)
        self.screen.blit(hand_label, (x - hand_label.get_width() // 2, y + r + 24))

    def _draw_ai_turn_indicator(self):
        label = self.font_small.render(f"{self._current_player()} is lining up...", True, GRAY)
        self.screen.blit(label, (24, 112))
        if self.ai_timer is not None:
            frac = max(0.0, min(1.0, self.ai_timer / self.AI_THINK_TIME))
            bar_w = 160
            pygame.draw.rect(self.screen, DARK_GRAY, (24, 144, bar_w, 10), border_radius=5)
            pygame.draw.rect(self.screen, YELLOW, (24, 144, int(bar_w * (1 - frac)), 10), border_radius=5)

    def _draw_scoreboard(self):
        hole_text = self.font_small.render(f"HOLE {self.hole_idx + 1} / {self.HOLES_PER_ROUND}", True, GRAY)
        self.screen.blit(hole_text, (WINDOW_W // 2 - hole_text.get_width() // 2, 8))

        def total_for(p):
            return sum(self.scores[p]) + (self.strokes if p == self._current_player() else 0)

        if len(self.players) == 1:
            p = self.players[0]
            score_text = self.font_big.render(f"Strokes: {self.strokes}   Total: {total_for(p)}", True, WHITE)
            self.screen.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 30))
        else:
            p1, p2 = self.players
            color1 = WHITE if self._current_player() == p1 else GRAY
            color2 = WHITE if self._current_player() == p2 else GRAY
            p1_text = self.font_med.render(f"{p1}: {total_for(p1)}", True, color1)
            p2_text = self.font_med.render(f"{p2}: {total_for(p2)}", True, color2)
            gap = 40
            total_w = p1_text.get_width() + p2_text.get_width() + gap
            start_x = WINDOW_W // 2 - total_w // 2
            self.screen.blit(p1_text, (start_x, 30))
            self.screen.blit(p2_text, (start_x + p1_text.get_width() + gap, 30))

    def _draw_hud(self):
        yards_left = abs(self.hole_x - self.ball_x) * self.DISTANCE_UNIT_SCALE
        info_text = self.font_small.render(
            f"{self.difficulty}  -  Par {self.par}  -  {yards_left:0.0f} yd to hole", True, GRAY)
        self.screen.blit(info_text, (WINDOW_W - info_text.get_width() - 20, 20))

        if self.obstacle_type == "wind" and self.wind_accel:
            arrow = "→" if self.wind_accel > 0 else "←"
            wind_text = self.font_small.render(f"WIND {arrow}", True, ACCENT)
            self.screen.blit(wind_text, (WINDOW_W - wind_text.get_width() - 20, 46))

        warn_y = WINDOW_H // 2 - 40
        if not self._current_player_is_ai():
            if not self.controller.get_right_detected():
                warn = self.font_small.render("RIGHT hand not detected -- needed for power", True, RED)
                self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, warn_y))
                warn_y += 26
            if not self.controller.get_left_detected():
                warn = self.font_small.render("LEFT hand not detected -- needed to shoot", True, RED)
                self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, warn_y))

        if self._current_player_is_ai():
            hint_str = f"{self._current_player()} is taking their shot... | R = restart match | ESC = menu"
        else:
            hint_str = (f"{self._current_player()}'s turn  -  RIGHT thumb = power | LEFT thumb UP = shoot "
                        "| R = restart match | ESC = menu")
        hint = self.font_small.render(hint_str, True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 20))

    def _draw_turn_complete_overlay(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        if self.turn_result_holed:
            text, color = f"{self.turn_result_player} holed in {self.turn_result_strokes}!", GREEN
        else:
            text, color = f"{self.turn_result_player} picks up after {self.turn_result_strokes} strokes", YELLOW
        msg = self.font_big.render(text, True, color)
        self.screen.blit(msg, (WINDOW_W // 2 - msg.get_width() // 2, WINDOW_H // 2 - 20))

    def _draw_round_complete_overlay(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title = self.font_big.render("ROUND COMPLETE!", True, GREEN)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, WINDOW_H // 2 - 100))

        totals = {p: sum(self.scores[p]) for p in self.players}
        if len(self.players) == 1:
            p = self.players[0]
            line = self.font_med.render(f"Total strokes: {totals[p]}", True, WHITE)
            self.screen.blit(line, (WINDOW_W // 2 - line.get_width() // 2, WINDOW_H // 2 - 30))
        else:
            p1, p2 = self.players
            line = self.font_med.render(f"{p1}: {totals[p1]}   {p2}: {totals[p2]}", True, WHITE)
            self.screen.blit(line, (WINDOW_W // 2 - line.get_width() // 2, WINDOW_H // 2 - 30))
            if totals[p1] < totals[p2]:
                winner = f"{p1} wins!"
            elif totals[p2] < totals[p1]:
                winner = f"{p2} wins!"
            else:
                winner = "It's a tie!"
            wtext = self.font_med.render(winner, True, YELLOW)
            self.screen.blit(wtext, (WINDOW_W // 2 - wtext.get_width() // 2, WINDOW_H // 2 + 10))

        retry_text = self.font_small.render("Press R to play again, ESC for menu", True, WHITE)
        self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 60))


class ThumbCurlingGame(BaseGame):
    """Curling, played top-down across a 3-round match.

    Two-hand, two-PHASE control per shot -- LEFT thumb always fires the
    "confirm" gesture (a thumbs-up), RIGHT thumb supplies the value being
    confirmed:
        1. RIGHT thumb aims the stone (live direction indicator).
        2. LEFT thumb up -> locks in that direction, moves to power.
        3. RIGHT thumb sets power (live power meter).
        4. LEFT thumb up -> locks in power AND releases the stone.
    The stone slides in a straight line at the chosen angle/speed and
    gradually decelerates on the ice, coming to rest somewhere on the
    sheet -- score is based on which ring of the house it lands in.

    Modes: 1 Player, 2 Player (alternate throws on the same round before
    advancing), or vs AI (a deliberately imprecise opponent). Each
    difficulty has its own set of 3 house placements -- Hard's house sits
    further off-center and adds a guard stone to navigate around."""

    name = "Thumb Curling"
    SUPPORTS_DIFFICULTY = False  # handled internally, same reasoning as Thumb Golf
    SUPPORTS_TWO_PLAYER = False

    SHEET_LEFT = 220
    SHEET_RIGHT = WINDOW_W - 220
    SHEET_TOP = 90
    SHEET_BOTTOM = WINDOW_H - 40
    SHEET_CENTER_X = (SHEET_LEFT + SHEET_RIGHT) / 2

    STONE_START_X = SHEET_CENTER_X
    STONE_START_Y = SHEET_BOTTOM - 30
    HOUSE_Y = SHEET_TOP + 70

    STONE_RADIUS = 12
    GUARD_RADIUS = 14

    MAX_ANGLE = math.radians(22)
    FRICTION = 250.0     # px/sec^2 -- ice deceleration
    MIN_SPEED = 60.0
    MAX_SPEED = 460.0
    STOP_SPEED = 2.0

    SHOOT_THRESHOLD = 0.75   # LEFT thumb value above this counts as a "confirm" gesture

    RING_RADII_POINTS = [(18, 5), (36, 3), (54, 2), (72, 1)]  # base radii (scaled by house_scale), outer to inner order not required

    ROUNDS_PER_MATCH = 3
    TURN_TRANSITION_TIME = 1.8

    AI_THINK_TIME = 1.0
    AI_DIRECTION_NOISE = 0.16   # fraction of the 0-1 control range -- this IS the bad AI
    AI_POWER_NOISE = 0.20

    # 3 house positions per difficulty (one per round) -- horizontal offset
    # in px from the sheet's center line. Harder difficulties push the
    # house further off-center, shrink the scoring rings, and (Hard only)
    # add a guard stone to navigate around.
    DIFFICULTY_SETTINGS = {
        "Easy":   {"house_offsets": [0, 24, -24], "house_scale": 1.15, "guard": False},
        "Medium": {"house_offsets": [54, -45, 66], "house_scale": 1.0, "guard": False},
        "Hard":   {"house_offsets": [84, -75, 60], "house_scale": 0.85, "guard": True},
    }

    def __init__(self, screen, controller, difficulty=None, mode="AI"):
        super().__init__(screen, controller, difficulty, mode)

        self.font_big = load_app_font("bold", 40)
        self.font_med = load_app_font("bold", 26)
        self.font_small = load_app_font("regular", 18)
        self.font_tiny = load_app_font("regular", 15)
        self._thumbsup_icon = render_icon("\U0001F44D", "UP", 22, self.font_tiny, WHITE)

        self._build_sheet_texture()

        self.match_mode = "1P"
        self.difficulty = "Medium"
        self.phase = "MODE_SELECT"
        self._build_mode_buttons()

    # ---------------------------------------------------------------
    # sheet art
    # ---------------------------------------------------------------

    def _build_sheet_texture(self):
        w = int(self.SHEET_RIGHT - self.SHEET_LEFT)
        h = int(self.SHEET_BOTTOM - self.SHEET_TOP)
        ice_top, ice_bottom = (150, 205, 230), (100, 170, 200)

        surf = pygame.Surface((w, h))
        for y in range(h):
            t = y / max(1, h - 1)
            color = tuple(int(ice_top[i] + (ice_bottom[i] - ice_top[i]) * t) for i in range(3))
            pygame.draw.line(surf, color, (0, y), (w, y))

        shine = pygame.Surface((w, h), pygame.SRCALPHA)
        for x in range(-h, w, 90):
            pygame.draw.line(shine, (255, 255, 255, 20), (x, 0), (x + h, h), 30)
        surf.blit(shine, (0, 0))

        # tee line + back line, purely decorative
        tee_y = int(self.HOUSE_Y - self.SHEET_TOP)
        pygame.draw.line(surf, (230, 240, 248), (0, tee_y), (w, tee_y), 2)

        self._sheet_surface = surf

    # ---------------------------------------------------------------
    # mode / difficulty select (self-contained, same pattern as Thumb Golf)
    # ---------------------------------------------------------------

    def _build_mode_buttons(self):
        btn_w, btn_h, gap = 320, 64, 20
        total_h = 3 * btn_h + 2 * gap
        start_y = WINDOW_H // 2 - total_h // 2 + 20
        specs = [
            ("1P", "1 Player", "\U0001F464", "1"),
            ("2P", "2 Player", "\U0001F465", "2"),
            ("AI", "vs AI", "\U0001F916", "A"),
        ]
        self.mode_buttons = []
        for i, (mode_id, label, icon, fallback) in enumerate(specs):
            rect = pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y + i * (btn_h + gap), btn_w, btn_h)
            btn = Button(rect, label, self.font_med, icon_emoji=icon, icon_fallback=fallback,
                         base_color=GREEN, text_color=BLACK)
            self.mode_buttons.append((btn, mode_id))

    def _build_difficulty_buttons(self):
        btn_w, btn_h, gap = 320, 64, 20
        total_h = 3 * btn_h + 2 * gap
        start_y = WINDOW_H // 2 - total_h // 2 + 20
        specs = [("Easy", GREEN), ("Medium", YELLOW), ("Hard", RED)]
        self.difficulty_buttons = []
        for i, (label, color) in enumerate(specs):
            rect = pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y + i * (btn_h + gap), btn_w, btn_h)
            btn = Button(rect, label, self.font_med, base_color=color, text_color=BLACK)
            self.difficulty_buttons.append((btn, label))
        self.back_button = Button(pygame.Rect(30, 30, 110, 46), "Back", self.font_small,
                                   icon_fallback="←", base_color=DARK_GRAY, text_color=WHITE)

    def _draw_mode_select(self):
        self.screen.blit(BACKGROUND, (0, 0))
        title = self.font_big.render("Thumb Curling", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 90))
        subtitle = self.font_small.render("Choose how you want to play", True, GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 150))
        for btn, _ in self.mode_buttons:
            btn.draw(self.screen)

    def _draw_difficulty_select(self):
        self.screen.blit(BACKGROUND, (0, 0))
        title = self.font_big.render("Choose Difficulty", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 90))
        subtitle = self.font_small.render(
            "3 rounds each -- harder difficulties push the house off-center", True, GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 150))
        for btn, _ in self.difficulty_buttons:
            btn.draw(self.screen)
        self.back_button.draw(self.screen)

    # ---------------------------------------------------------------
    # match / turn / round setup
    # ---------------------------------------------------------------

    def start_match(self):
        if self.match_mode == "1P":
            self.players = ["YOU"]
        elif self.match_mode == "2P":
            self.players = ["P1", "P2"]
        else:
            self.players = ["YOU", "AI"]
        self.scores = {p: [] for p in self.players}
        self.round_idx = 0
        self._setup_round()
        self.phase = "PLAYING"

    def _setup_round(self):
        settings = self.DIFFICULTY_SETTINGS[self.difficulty]
        offset = settings["house_offsets"][self.round_idx]
        self.house_x = self.SHEET_CENTER_X + offset
        self.house_scale = settings["house_scale"]
        self.guard_active = settings["guard"]
        self.guard_x = self.SHEET_CENTER_X
        self.guard_y = self.STONE_START_Y - 0.65 * (self.STONE_START_Y - self.HOUSE_Y)
        self.turn_idx = 0
        self._start_player_turn()

    def _start_player_turn(self):
        self.ball_x = float(self.STONE_START_X)
        self.ball_y = float(self.STONE_START_Y)
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_state = "ready"
        self.shot_phase = "AIMING"
        self.direction_live = 0.5
        self.power_live = 0.0
        self.chosen_direction = None
        self.chosen_power = None
        self.removed_from_play = False
        self._prev_left_value = None
        self.ai_timer = None

    def _current_player(self):
        return self.players[self.turn_idx]

    def _current_player_is_ai(self):
        return self._current_player() == "AI"

    def handle_event(self, event):
        if self.phase == "MODE_SELECT":
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, mode_id in self.mode_buttons:
                    if btn.is_clicked(event.pos):
                        self.match_mode = mode_id
                        self._build_difficulty_buttons()
                        self.phase = "DIFFICULTY_SELECT"
                        return
        elif self.phase == "DIFFICULTY_SELECT":
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, label in self.difficulty_buttons:
                    if btn.is_clicked(event.pos):
                        self.difficulty = label
                        self.start_match()
                        return
                if self.back_button.is_clicked(event.pos):
                    self._build_mode_buttons()
                    self.phase = "MODE_SELECT"
                    return
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._build_mode_buttons()
                self.phase = "MODE_SELECT"

    # ---------------------------------------------------------------
    # update
    # ---------------------------------------------------------------

    def update(self, dt):
        if self.phase in ("MODE_SELECT", "DIFFICULTY_SELECT"):
            mouse_pos = pygame.mouse.get_pos()
            mouse_down = pygame.mouse.get_pressed()[0]
            buttons = self.mode_buttons if self.phase == "MODE_SELECT" else self.difficulty_buttons
            for btn, _ in buttons:
                btn.update(dt, mouse_pos, mouse_down)
            if self.phase == "DIFFICULTY_SELECT":
                self.back_button.update(dt, mouse_pos, mouse_down)
        elif self.phase == "PLAYING":
            self._update_playing(dt)
        elif self.phase == "TURN_COMPLETE":
            self.turn_transition_timer -= dt
            if self.turn_transition_timer <= 0:
                self._advance_after_turn()
        # MATCH_COMPLETE: nothing to update, just waiting for R

    def _update_playing(self, dt):
        if self.ball_state == "ready":
            if self._current_player_is_ai():
                if self.ai_timer is None:
                    self.ai_timer = self.AI_THINK_TIME
                else:
                    self.ai_timer -= dt
                    if self.ai_timer <= 0:
                        self._take_ai_shot()
            else:
                self._update_human_shot(dt)
        elif self.ball_state == "sliding":
            self._update_slide(dt)
            if self.ball_state == "stopped":
                self._end_turn()

    def _update_human_shot(self, dt):
        right_value = max(0.0, min(1.0, self.controller.get_right_value()))
        left_value = max(0.0, min(1.0, self.controller.get_left_value()))

        if self._prev_left_value is None:
            self._prev_left_value = left_value
            rising_edge = False
        else:
            rising_edge = self._prev_left_value <= self.SHOOT_THRESHOLD < left_value
            self._prev_left_value = left_value

        if self.shot_phase == "AIMING":
            self.direction_live = right_value
            if rising_edge:
                self.chosen_direction = right_value
                self.shot_phase = "POWER"
        elif self.shot_phase == "POWER":
            self.power_live = right_value
            if rising_edge:
                self.chosen_power = right_value
                self._launch_stone(self.chosen_direction, self.chosen_power)

    def _estimate_good_shot(self):
        dx = self.house_x - self.STONE_START_X
        dy = self.HOUSE_Y - self.STONE_START_Y  # negative -- house is up-sheet from the start
        target_distance = math.hypot(dx, dy)
        target_angle = math.atan2(dx, -dy)
        direction_value = 0.5 + (target_angle / self.MAX_ANGLE) * 0.5
        required_speed = math.sqrt(max(0.0, 2 * self.FRICTION * target_distance))
        power_value = (required_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        return max(0.0, min(1.0, direction_value)), max(0.0, min(1.0, power_value))

    def _take_ai_shot(self):
        target_direction, target_power = self._estimate_good_shot()
        noisy_direction = max(0.0, min(1.0, target_direction + random.uniform(-self.AI_DIRECTION_NOISE, self.AI_DIRECTION_NOISE)))
        noisy_power = max(0.0, min(1.0, target_power + random.uniform(-self.AI_POWER_NOISE, self.AI_POWER_NOISE)))
        self.chosen_direction = noisy_direction
        self.chosen_power = noisy_power
        self._launch_stone(noisy_direction, noisy_power)
        self.ai_timer = None

    def _launch_stone(self, direction_value, power_value):
        angle = (direction_value - 0.5) * 2 * self.MAX_ANGLE
        speed = self.MIN_SPEED + power_value * (self.MAX_SPEED - self.MIN_SPEED)
        self.ball_vx = speed * math.sin(angle)
        self.ball_vy = -speed * math.cos(angle)
        self.ball_state = "sliding"
        self.removed_from_play = False

    def _update_slide(self, dt):
        speed = math.hypot(self.ball_vx, self.ball_vy)
        if speed <= 0:
            self.ball_state = "stopped"
            return

        new_speed = max(0.0, speed - self.FRICTION * dt)
        scale = new_speed / speed
        self.ball_vx *= scale
        self.ball_vy *= scale
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        if self.guard_active:
            dist_to_guard = math.hypot(self.ball_x - self.guard_x, self.ball_y - self.guard_y)
            if dist_to_guard < self.STONE_RADIUS + self.GUARD_RADIUS:
                self.ball_state = "stopped"
                return

        if (self.ball_x < self.SHEET_LEFT or self.ball_x > self.SHEET_RIGHT
                or self.ball_y < self.SHEET_TOP - 20):
            self.ball_state = "stopped"
            self.removed_from_play = True
            return

        if new_speed < self.STOP_SPEED:
            self.ball_vx = self.ball_vy = 0.0
            self.ball_state = "stopped"

    def _points_for_distance(self, dist):
        for radius, points in self.RING_RADII_POINTS:
            if dist <= radius * self.house_scale:
                return points
        return 0

    def _end_turn(self):
        if self.removed_from_play:
            points = 0
        else:
            dist = math.hypot(self.ball_x - self.house_x, self.ball_y - self.HOUSE_Y)
            points = self._points_for_distance(dist)

        player = self._current_player()
        self.scores[player].append(points)
        self.turn_result_player = player
        self.turn_result_points = points
        self.turn_result_removed = self.removed_from_play
        self.phase = "TURN_COMPLETE"
        self.turn_transition_timer = self.TURN_TRANSITION_TIME

    def _advance_after_turn(self):
        self.turn_idx += 1
        if self.turn_idx >= len(self.players):
            self.turn_idx = 0
            self.round_idx += 1
            if self.round_idx >= self.ROUNDS_PER_MATCH:
                self.phase = "MATCH_COMPLETE"
                return
            self._setup_round()
        else:
            self._start_player_turn()
        self.phase = "PLAYING"

    # ---------------------------------------------------------------
    # draw
    # ---------------------------------------------------------------

    def draw(self):
        if self.phase == "MODE_SELECT":
            self._draw_mode_select()
        elif self.phase == "DIFFICULTY_SELECT":
            self._draw_difficulty_select()
        elif self.phase in ("PLAYING", "TURN_COMPLETE"):
            self._draw_playing()
            if self.phase == "TURN_COMPLETE":
                self._draw_turn_complete_overlay()
        elif self.phase == "MATCH_COMPLETE":
            self._draw_playing()
            self._draw_match_complete_overlay()

    def _draw_playing(self):
        self.screen.blit(BACKGROUND, (0, 0))
        self.screen.blit(self._sheet_surface, (self.SHEET_LEFT, self.SHEET_TOP))

        self._draw_house()
        if self.guard_active:
            pygame.draw.circle(self.screen, GRAY, (int(self.guard_x), int(self.guard_y)), self.GUARD_RADIUS)
            pygame.draw.circle(self.screen, DARK_GRAY, (int(self.guard_x), int(self.guard_y)), self.GUARD_RADIUS, 2)

        self._draw_aim_indicator()

        stone_color = RED if self._current_player() in ("P2", "AI") else BLUE
        pygame.draw.circle(self.screen, stone_color, (int(self.ball_x), int(self.ball_y)), self.STONE_RADIUS)
        pygame.draw.circle(self.screen, DARK_GRAY, (int(self.ball_x), int(self.ball_y)), self.STONE_RADIUS, 2)

        if self._current_player_is_ai():
            self._draw_ai_turn_indicator()
        else:
            self._draw_control_meters()

        self._draw_scoreboard()
        self._draw_hud()

    def _draw_house(self):
        rings = [(72, RED), (54, WHITE), (36, RED), (18, BLUE)]
        for radius, color in rings:
            pygame.draw.circle(self.screen, color, (int(self.house_x), int(self.HOUSE_Y)),
                                int(radius * self.house_scale))

    def _draw_aim_indicator(self):
        if self.ball_state != "ready" or self._current_player_is_ai():
            return
        if self.shot_phase == "AIMING":
            direction_value, live = self.direction_live, True
        else:
            direction_value, live = self.chosen_direction, False
        if direction_value is None:
            return
        angle = (direction_value - 0.5) * 2 * self.MAX_ANGLE
        end_x = self.ball_x + 150 * math.sin(angle)
        end_y = self.ball_y - 150 * math.cos(angle)
        color = ACCENT if live else WHITE
        pygame.draw.line(self.screen, color, (self.ball_x, self.ball_y), (end_x, end_y), 3)

    def _draw_control_meters(self):
        # direction meter (horizontal bar) -- active during AIMING
        x, y, w, h = WINDOW_W // 2 - 110, 96, 220, 16
        active_dir = self.shot_phase == "AIMING"
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, w, h), border_radius=8)
        pygame.draw.line(self.screen, GRAY, (x + w // 2, y - 4), (x + w // 2, y + h + 4), 2)
        dir_value = self.direction_live if active_dir else (self.chosen_direction or 0.5)
        knob_x = x + int(dir_value * w)
        knob_color = ACCENT if active_dir else GRAY
        pygame.draw.circle(self.screen, knob_color, (knob_x, y + h // 2), 11)
        pygame.draw.circle(self.screen, WHITE, (knob_x, y + h // 2), 11, 2)
        label = self.font_tiny.render("DIRECTION (RIGHT thumb)", True, GRAY)
        self.screen.blit(label, (x + w // 2 - label.get_width() // 2, y - 20))

        # power meter (vertical bar) -- active during POWER, same style as Thumb Golf's
        px, py, pw, ph = 24, 130, 26, 200
        pygame.draw.rect(self.screen, DARK_GRAY, (px, py, pw, ph), border_radius=8)
        active_power = self.shot_phase == "POWER"
        fill_h = int(ph * self.power_live) if active_power else 0
        if fill_h > 0:
            fill_color = GREEN if self.power_live < 0.55 else (YELLOW if self.power_live < 0.82 else RED)
            pygame.draw.rect(self.screen, fill_color, (px, py + ph - fill_h, pw, fill_h), border_radius=8)
        pygame.draw.rect(self.screen, GRAY if not active_power else GREEN, (px, py, pw, ph), width=2, border_radius=8)
        plabel = self.font_tiny.render("POWER", True, GRAY)
        self.screen.blit(plabel, (px + pw // 2 - plabel.get_width() // 2, py - 20))
        hand_label = self.font_tiny.render("RIGHT", True, GRAY)
        self.screen.blit(hand_label, (px + pw // 2 - hand_label.get_width() // 2, py + ph + 6))

        # shoot indicator (left hand) -- shows whether the confirm gesture is currently "up"
        sx, sy, r = 37, 372, 20
        ready = self._prev_left_value is not None and self._prev_left_value > self.SHOOT_THRESHOLD
        color = GREEN if ready else DARK_GRAY
        pygame.draw.circle(self.screen, color, (sx, sy), r)
        pygame.draw.circle(self.screen, GRAY, (sx, sy), r, 2)
        icon = self._thumbsup_icon
        self.screen.blit(icon, (sx - icon.get_width() // 2, sy - icon.get_height() // 2))
        label = self.font_tiny.render("CONFIRM", True, GRAY)
        self.screen.blit(label, (sx - label.get_width() // 2, sy + r + 6))
        hand_label2 = self.font_tiny.render("LEFT", True, GRAY)
        self.screen.blit(hand_label2, (sx - hand_label2.get_width() // 2, sy + r + 24))

    def _draw_ai_turn_indicator(self):
        label = self.font_small.render(f"{self._current_player()} is lining up...", True, GRAY)
        self.screen.blit(label, (24, 112))
        if self.ai_timer is not None:
            frac = max(0.0, min(1.0, self.ai_timer / self.AI_THINK_TIME))
            bar_w = 160
            pygame.draw.rect(self.screen, DARK_GRAY, (24, 144, bar_w, 10), border_radius=5)
            pygame.draw.rect(self.screen, YELLOW, (24, 144, int(bar_w * (1 - frac)), 10), border_radius=5)

    def _draw_scoreboard(self):
        round_text = self.font_small.render(f"ROUND {self.round_idx + 1} / {self.ROUNDS_PER_MATCH}", True, GRAY)
        self.screen.blit(round_text, (WINDOW_W // 2 - round_text.get_width() // 2, 8))

        def total_for(p):
            return sum(self.scores[p])

        if len(self.players) == 1:
            p = self.players[0]
            score_text = self.font_big.render(f"Score: {total_for(p)}", True, WHITE)
            self.screen.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 30))
        else:
            p1, p2 = self.players
            color1 = WHITE if self._current_player() == p1 else GRAY
            color2 = WHITE if self._current_player() == p2 else GRAY
            p1_text = self.font_med.render(f"{p1}: {total_for(p1)}", True, color1)
            p2_text = self.font_med.render(f"{p2}: {total_for(p2)}", True, color2)
            gap = 40
            total_w = p1_text.get_width() + p2_text.get_width() + gap
            start_x = WINDOW_W // 2 - total_w // 2
            self.screen.blit(p1_text, (start_x, 30))
            self.screen.blit(p2_text, (start_x + p1_text.get_width() + gap, 30))

    def _draw_hud(self):
        info_text = self.font_small.render(f"{self.difficulty} difficulty", True, GRAY)
        self.screen.blit(info_text, (WINDOW_W - info_text.get_width() - 20, 20))

        warn_y = WINDOW_H // 2 - 40
        if not self._current_player_is_ai():
            if not self.controller.get_right_detected():
                warn = self.font_small.render("RIGHT hand not detected -- needed to aim/power", True, RED)
                self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, warn_y))
                warn_y += 26
            if not self.controller.get_left_detected():
                warn = self.font_small.render("LEFT hand not detected -- needed to confirm", True, RED)
                self.screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, warn_y))

        if self._current_player_is_ai():
            hint_str = f"{self._current_player()} is taking their shot... | R = restart match | ESC = menu"
        elif self.shot_phase == "AIMING":
            hint_str = f"{self._current_player()}'s turn  -  RIGHT thumb = aim | LEFT thumb UP = confirm direction"
        else:
            hint_str = f"{self._current_player()}'s turn  -  RIGHT thumb = power | LEFT thumb UP = shoot"
        hint = self.font_small.render(hint_str, True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 20))

    def _draw_turn_complete_overlay(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        if self.turn_result_removed:
            text, color = f"{self.turn_result_player}'s stone went out of play!", RED
        elif self.turn_result_points > 0:
            text, color = f"{self.turn_result_player} scores {self.turn_result_points} point(s)!", GREEN
        else:
            text, color = f"{self.turn_result_player}'s stone missed the house", YELLOW
        msg = self.font_big.render(text, True, color)
        self.screen.blit(msg, (WINDOW_W // 2 - msg.get_width() // 2, WINDOW_H // 2 - 20))

    def _draw_match_complete_overlay(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title = self.font_big.render("MATCH COMPLETE!", True, GREEN)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, WINDOW_H // 2 - 100))

        totals = {p: sum(self.scores[p]) for p in self.players}
        if len(self.players) == 1:
            p = self.players[0]
            line = self.font_med.render(f"Total score: {totals[p]}", True, WHITE)
            self.screen.blit(line, (WINDOW_W // 2 - line.get_width() // 2, WINDOW_H // 2 - 30))
        else:
            p1, p2 = self.players
            line = self.font_med.render(f"{p1}: {totals[p1]}   {p2}: {totals[p2]}", True, WHITE)
            self.screen.blit(line, (WINDOW_W // 2 - line.get_width() // 2, WINDOW_H // 2 - 30))
            if totals[p1] > totals[p2]:
                winner = f"{p1} wins!"
            elif totals[p2] > totals[p1]:
                winner = f"{p2} wins!"
            else:
                winner = "It's a tie!"
            wtext = self.font_med.render(winner, True, YELLOW)
            self.screen.blit(wtext, (WINDOW_W // 2 - wtext.get_width() // 2, WINDOW_H // 2 + 10))

        retry_text = self.font_small.render("Press R to play again, ESC for menu", True, WHITE)
        self.screen.blit(retry_text, (WINDOW_W // 2 - retry_text.get_width() // 2, WINDOW_H // 2 + 60))


# ============================================================
# GAME REGISTRY -- add new games here
# ============================================================

GAMES = {
    "Air Hockey": {"class": AirHockeyGame, "icon": "🏒", "fallback_icon": "o"},
    "Balance Beam": {"class": BalanceBeamGame, "icon": "⚖", "fallback_icon": "="},
    "Breakout": {"class": BreakoutGame, "icon": "🧱", "fallback_icon": "▪"},
    "Thumb Golf": {"class": ThumbGolfGame, "icon": "⛳", "fallback_icon": "P"},
    "Thumb Curling": {"class": ThumbCurlingGame, "icon": "🥌", "fallback_icon": "C"},
    "Pong": {"class": PongGame, "icon": "\U0001F3AE", "fallback_icon": "\u25B6"},  # 🎮 / ▶
    # "NextGame": {"class": NextGameClass, "icon": "\U0001F3B2", "fallback_icon": "\u2666"},
}


# ============================================================
# MENU SCREEN
# ============================================================

class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font_title = load_app_font("bold", 46)
        self.font_button = load_app_font("bold", 30)
        self.font_small = load_app_font("regular", 20)

        # Lay buttons out in the space between the title panel and the
        # bottom status hints, shrinking button height/gap if there are
        # enough games that they wouldn't otherwise fit -- fixes buttons
        # running off the bottom of the window as more games get added.
        top_limit = 225
        bottom_limit = WINDOW_H - 55
        available_h = bottom_limit - top_limit

        n = len(GAMES)
        button_h, gap = 68, 18
        total_h = n * button_h + (n - 1) * gap
        if total_h > available_h and n > 0:
            scale = available_h / total_h
            button_h = max(46, int(button_h * scale))
            gap = max(8, int(gap * scale))
            total_h = n * button_h + (n - 1) * gap
        start_y = top_limit + max(0, (available_h - total_h) // 2)

        self.buttons = []
        for i, (name, info) in enumerate(GAMES.items()):
            rect = pygame.Rect(WINDOW_W // 2 - 170, start_y + i * (button_h + gap), 340, button_h)
            btn = Button(
                rect, name, self.font_button,
                icon_emoji=info.get("icon"), icon_fallback=info.get("fallback_icon"),
                base_color=GREEN, text_color=BLACK,
            )
            self.buttons.append((btn, name))

    def update(self, dt, mouse_pos, mouse_down):
        for btn, _ in self.buttons:
            btn.update(dt, mouse_pos, mouse_down)

    def handle_click(self, pos):
        for btn, name in self.buttons:
            if btn.is_clicked(pos):
                return name
        return None

    def draw(self, hand_detected):
        self.screen.blit(BACKGROUND, (0, 0))

        # title card -- rounded square-ish panel behind the title
        panel_rect = pygame.Rect(0, 0, 420, 130)
        panel_rect.center = (WINDOW_W // 2, 150)
        panel_surf = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (255, 255, 255, 14), panel_surf.get_rect(), border_radius=30)
        pygame.draw.rect(panel_surf, (*ACCENT, 130), panel_surf.get_rect(), width=3, border_radius=30)
        self.screen.blit(panel_surf, panel_rect.topleft)

        title = self.font_title.render("Thumb Games", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2,
                                  panel_rect.centery - title.get_height() // 2))

        for btn, _ in self.buttons:
            btn.draw(self.screen)

        status = "Hand detected" if hand_detected else "No hand detected -- show your thumb"
        color = GREEN if hand_detected else RED
        status_text = self.font_small.render(status, True, color)
        self.screen.blit(status_text, (20, WINDOW_H - 30))

        quit_hint = self.font_small.render("Press Q to quit", True, GRAY)
        self.screen.blit(quit_hint, (WINDOW_W - quit_hint.get_width() - 20, WINDOW_H - 30))


# ============================================================
# MODE SELECT SCREEN (1 Player vs AI, or 2 Player)
# ============================================================

class ModeSelect:
    def __init__(self, screen, game_name):
        self.screen = screen
        self.game_name = game_name

        self.font_title = load_app_font("bold", 42)
        self.font_button = load_app_font("bold", 28)
        self.font_small = load_app_font("regular", 20)

        btn_w, btn_h = 340, 70
        gap = 24
        total_h = btn_h * 2 + gap
        start_y = WINDOW_H // 2 - total_h // 2 + 10

        self.one_player_btn = Button(
            pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y, btn_w, btn_h),
            "1 Player (vs AI)", self.font_button,
            icon_emoji="\U0001F916", icon_fallback="1",
            base_color=GREEN, text_color=BLACK,
        )
        self.two_player_btn = Button(
            pygame.Rect(WINDOW_W // 2 - btn_w // 2, start_y + btn_h + gap, btn_w, btn_h),
            "2 Player", self.font_button,
            icon_emoji="\U0001F465", icon_fallback="2",
            base_color=GREEN, text_color=BLACK,
        )
        self.back_button = Button(
            pygame.Rect(30, 30, 110, 46), "Back", self.font_small,
            icon_fallback="\u2190", base_color=DARK_GRAY, text_color=WHITE,
        )

    def update(self, dt, mouse_pos, mouse_down):
        self.one_player_btn.update(dt, mouse_pos, mouse_down)
        self.two_player_btn.update(dt, mouse_pos, mouse_down)
        self.back_button.update(dt, mouse_pos, mouse_down)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "BACK"

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.one_player_btn.is_clicked(event.pos):
                return "ONE_PLAYER"
            if self.two_player_btn.is_clicked(event.pos):
                return "TWO_PLAYER"
            if self.back_button.is_clicked(event.pos):
                return "BACK"

        return None

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))

        title = self.font_title.render(f"{self.game_name} -- Choose Mode", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 150))

        self.one_player_btn.draw(self.screen)
        self.two_player_btn.draw(self.screen)
        self.back_button.draw(self.screen)

        hint = self.font_small.render("2 Player: left hand controls the left paddle, right hand the right paddle", True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 60))


# ============================================================
# DIFFICULTY SELECT SCREEN (slider with 3 snap positions)
# ============================================================

class DifficultySelect:
    LEVELS = ["Easy", "Medium", "Hard"]
    LEVEL_COLORS = [GREEN, YELLOW, RED]

    TRACK_LEFT = WINDOW_W // 2 - 200
    TRACK_RIGHT = WINDOW_W // 2 + 200
    TRACK_Y = 320
    TRACK_HEIGHT = 8
    HANDLE_RADIUS = 16

    def __init__(self, screen, game_name, initial_index=1):
        self.screen = screen
        self.game_name = game_name
        self.selected_index = initial_index
        self.dragging = False

        self.font_title = load_app_font("bold", 40)
        self.font_label = load_app_font("regular", 26)
        self.font_small = load_app_font("regular", 20)
        self.font_button = load_app_font("bold", 28)

        self.snap_positions = [self.TRACK_LEFT, (self.TRACK_LEFT + self.TRACK_RIGHT) // 2, self.TRACK_RIGHT]

        self.start_button = Button(
            pygame.Rect(WINDOW_W // 2 - 110, 430, 220, 60), "Start", self.font_button,
            icon_fallback="\u25B6", base_color=GREEN, text_color=BLACK,
        )
        self.back_button = Button(
            pygame.Rect(30, 30, 110, 46), "Back", self.font_small,
            icon_fallback="\u2190", base_color=DARK_GRAY, text_color=WHITE,
        )

    def _handle_pos(self):
        return self.snap_positions[self.selected_index], self.TRACK_Y

    def _nearest_index(self, x):
        distances = [abs(x - pos) for pos in self.snap_positions]
        return distances.index(min(distances))

    def update(self, dt, mouse_pos, mouse_down):
        self.start_button.update(dt, mouse_pos, mouse_down)
        self.back_button.update(dt, mouse_pos, mouse_down)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "BACK"

        if event.type == pygame.MOUSEBUTTONDOWN:
            hx, hy = self._handle_pos()
            handle_rect = pygame.Rect(0, 0, self.HANDLE_RADIUS * 2, self.HANDLE_RADIUS * 2)
            handle_rect.center = (hx, hy)

            if handle_rect.collidepoint(event.pos):
                self.dragging = True
            elif (self.TRACK_LEFT - 20 <= event.pos[0] <= self.TRACK_RIGHT + 20
                  and abs(event.pos[1] - self.TRACK_Y) <= 20):
                self.selected_index = self._nearest_index(event.pos[0])
            elif self.start_button.is_clicked(event.pos):
                return "START"
            elif self.back_button.is_clicked(event.pos):
                return "BACK"

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.selected_index = self._nearest_index(event.pos[0])

        return None

    def get_difficulty(self):
        return self.LEVELS[self.selected_index]

    def draw(self):
        self.screen.blit(BACKGROUND, (0, 0))

        title = self.font_title.render(f"{self.game_name} -- Choose Difficulty", True, WHITE)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 140))

        track_rect = pygame.Rect(self.TRACK_LEFT, self.TRACK_Y - self.TRACK_HEIGHT // 2,
                                  self.TRACK_RIGHT - self.TRACK_LEFT, self.TRACK_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, track_rect, border_radius=4)

        for i, (pos, level, color) in enumerate(zip(self.snap_positions, self.LEVELS, self.LEVEL_COLORS)):
            pygame.draw.circle(self.screen, DARK_GRAY, (pos, self.TRACK_Y), 5)
            label = self.font_label.render(level, True, color if i == self.selected_index else GRAY)
            self.screen.blit(label, (pos - label.get_width() // 2, self.TRACK_Y + 30))

        hx, hy = self._handle_pos()
        handle_color = self.LEVEL_COLORS[self.selected_index]
        pygame.draw.circle(self.screen, handle_color, (hx, hy), self.HANDLE_RADIUS)
        pygame.draw.circle(self.screen, WHITE, (hx, hy), self.HANDLE_RADIUS, 2)

        self.start_button.draw(self.screen)
        self.back_button.draw(self.screen)

        hint = self.font_small.render("Drag the slider or click a position, then press Start", True, GRAY)
        self.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, 520))


# ============================================================
# LOADING SCREEN -- shown while the camera/model load in the background
# ============================================================

def draw_loading_screen(screen, font_title, font_small, message, submessage=None, color=WHITE):
    screen.blit(BACKGROUND, (0, 0))
    dots = "." * ((pygame.time.get_ticks() // 400) % 4)
    title = font_title.render(message + dots, True, color)
    screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, WINDOW_H // 2 - 40))
    if submessage:
        sub = font_small.render(submessage, True, GRAY)
        screen.blit(sub, (WINDOW_W // 2 - sub.get_width() // 2, WINDOW_H // 2 + 10))


# ============================================================
# MAIN APP LOOP
# ============================================================

BACKGROUND = None  # built once in main() after the display is created


def main():
    global BACKGROUND

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Thumb Games")
    pygame.display.set_icon(make_thumb_icon_surface(64))

    BACKGROUND = make_gradient(WINDOW_W, WINDOW_H, TOP_BG, BOTTOM_BG)

    clock = pygame.time.Clock()
    font_loading_title = load_app_font("bold", 32)
    font_loading_small = load_app_font("regular", 18)

    # Loading the hand-tracking model/camera pulls in PyTorch, MediaPipe and
    # OpenCV and can take several seconds. Do it in a background thread so
    # the window/menu show up immediately instead of sitting frozen.
    loader = {"controller": None, "error": None}

    def _load_controller():
        try:
            from thumb_control import ThumbController
            c = ThumbController()
            c.start()
            loader["controller"] = c
        except Exception as e:
            loader["error"] = str(e)

    threading.Thread(target=_load_controller, daemon=True).start()

    controller = None
    menu = None
    current_game = None
    mode_select = None
    difficulty_select = None
    pending_game_name = None
    state = "LOADING"  # "LOADING" -> "MENU" -> "MODE_SELECT" -> "DIFFICULTY" (if AI chosen) -> "PLAYING"

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = pygame.mouse.get_pressed()[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and state != "LOADING":
                    state = "MENU"
                    current_game = None
                    mode_select = None
                    difficulty_select = None
                elif event.key == pygame.K_q and state in ("MENU", "LOADING"):
                    running = False

            if state == "MENU" and event.type == pygame.MOUSEBUTTONDOWN:
                chosen = menu.handle_click(event.pos)
                if chosen is not None:
                    game_info = GAMES[chosen]
                    game_class = game_info["class"]
                    pending_game_name = chosen

                    supports_difficulty = getattr(game_class, "SUPPORTS_DIFFICULTY", False)
                    supports_two_player = getattr(game_class, "SUPPORTS_TWO_PLAYER", False)

                    if supports_difficulty and supports_two_player:
                        mode_select = ModeSelect(screen, chosen)
                        state = "MODE_SELECT"
                    elif supports_difficulty:
                        difficulty_select = DifficultySelect(screen, chosen)
                        state = "DIFFICULTY"
                    elif supports_two_player:
                        current_game = game_class(screen, controller, mode="2P")
                        state = "PLAYING"
                    else:
                        current_game = game_class(screen, controller)
                        state = "PLAYING"

            elif state == "MODE_SELECT" and mode_select is not None:
                result = mode_select.handle_event(event)
                game_class = GAMES[pending_game_name]["class"]
                if result == "ONE_PLAYER":
                    mode_select = None
                    difficulty_select = DifficultySelect(screen, pending_game_name)
                    state = "DIFFICULTY"
                elif result == "TWO_PLAYER":
                    current_game = game_class(screen, controller, mode="2P")
                    mode_select = None
                    state = "PLAYING"
                elif result == "BACK":
                    mode_select = None
                    state = "MENU"

            elif state == "DIFFICULTY" and difficulty_select is not None:
                result = difficulty_select.handle_event(event)
                if result == "START":
                    chosen_difficulty = difficulty_select.get_difficulty()
                    current_game = GAMES[pending_game_name]["class"](
                        screen, controller, difficulty=chosen_difficulty, mode="AI"
                    )
                    difficulty_select = None
                    state = "PLAYING"
                elif result == "BACK":
                    difficulty_select = None
                    # if this game also has a 2-player mode, go back to mode
                    # select rather than all the way to the main menu
                    game_class = GAMES[pending_game_name]["class"]
                    if getattr(game_class, "SUPPORTS_TWO_PLAYER", False):
                        mode_select = ModeSelect(screen, pending_game_name)
                        state = "MODE_SELECT"
                    else:
                        state = "MENU"

            elif state == "PLAYING" and current_game is not None:
                current_game.handle_event(event)

        if state == "LOADING":
            if loader["error"] is not None:
                draw_loading_screen(screen, font_loading_title, font_loading_small,
                                     "Failed to start camera/model", loader["error"], color=RED)
            elif loader["controller"] is not None:
                controller = loader["controller"]
                menu = Menu(screen)
                state = "MENU"
            else:
                draw_loading_screen(screen, font_loading_title, font_loading_small,
                                     "Loading camera & model",
                                     "This can take a few seconds the first time")
        elif state == "MENU":
            menu.update(dt, mouse_pos, mouse_down)
            menu.draw(controller.get_hand_detected())
        elif state == "MODE_SELECT" and mode_select is not None:
            mode_select.update(dt, mouse_pos, mouse_down)
            mode_select.draw()
        elif state == "DIFFICULTY" and difficulty_select is not None:
            difficulty_select.update(dt, mouse_pos, mouse_down)
            difficulty_select.draw()
        elif state == "PLAYING" and current_game is not None:
            current_game.update(dt)
            current_game.draw()

        pygame.display.flip()

    if controller is not None:
        controller.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()