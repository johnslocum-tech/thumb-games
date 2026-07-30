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
# GAME REGISTRY -- add new games here
# ============================================================

GAMES = {
    "Balance Beam": {"class": BalanceBeamGame, "icon": "⚖", "fallback_icon": "="},
    "Breakout": {"class": BreakoutGame, "icon": "🧱", "fallback_icon": "▪"},
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

        self.buttons = []
        start_y = 250
        spacing = 95
        for i, (name, info) in enumerate(GAMES.items()):
            rect = pygame.Rect(WINDOW_W // 2 - 170, start_y + i * spacing, 340, 68)
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