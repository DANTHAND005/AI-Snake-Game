"""
snake_ai/game.py
----------------
Snake game environment for the Deep Q-Learning agent.
Uses pygame for rendering. Call with render=False for headless training.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

# ── Constants ──────────────────────────────────────────────────────────────────
BLOCK   = 20          # pixels per grid cell
SPEED   = 40          # FPS (increase to train faster)
W, H    = 640, 480    # window size

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (200, 0,   0)
GREEN1 = (0,   200, 0)
GREEN2 = (0,   150, 0)
BLUE   = (0,   0,   200)
GRAY   = (40,  40,  40)

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4


class SnakeGame:
    """
    Snake environment compatible with the DQN agent.

    Action space (one-hot):
        [1, 0, 0] → go straight
        [0, 1, 0] → turn right
        [0, 0, 1] → turn left

    State (11 booleans):
        danger straight, danger right, danger left,
        moving left, moving right, moving up, moving down,
        food left, food right, food up, food down
    """

    def __init__(self, render: bool = True, w: int = W, h: int = H):
        self.w      = w
        self.h      = h
        self.render = render

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("AI Snake")
            self.clock   = pygame.time.Clock()
            self.font    = pygame.font.SysFont("monospace", 20)

        self.reset()

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        self.direction = Direction.RIGHT
        self.head      = Point(self.w // 2, self.h // 2)
        self.snake     = [
            self.head,
            Point(self.head.x - BLOCK, self.head.y),
            Point(self.head.x - 2 * BLOCK, self.head.y),
        ]
        self.score      = 0
        self.food       = None
        self.frame_iter = 0
        self._place_food()
        return self._get_state()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        action: list of 3 ints, one-hot encoded.
        Returns: state, reward, done, score
        """
        self.frame_iter += 1

        # Quit event
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        # Rewards
        reward = 0
        done   = False

        if self._is_collision() or self.frame_iter > 100 * len(self.snake):
            done   = True
            reward = -10
            return self._get_state(), reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._draw()

        return self._get_state(), reward, done, self.score

    # ── Collision ──────────────────────────────────────────────────────────────
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Wall
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Self
        if pt in self.snake[1:]:
            return True
        return False

    # ── State (11 values) ──────────────────────────────────────────────────────
    def _get_state(self):
        head = self.head
        d    = self.direction

        pt_l = Point(head.x - BLOCK, head.y)
        pt_r = Point(head.x + BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        dir_l = d == Direction.LEFT
        dir_r = d == Direction.RIGHT
        dir_u = d == Direction.UP
        dir_d = d == Direction.DOWN

        # Danger straight / right / left relative to current heading
        state = [
            # Danger straight
            (dir_r and self._is_collision(pt_r)) or
            (dir_l and self._is_collision(pt_l)) or
            (dir_u and self._is_collision(pt_u)) or
            (dir_d and self._is_collision(pt_d)),

            # Danger right
            (dir_u and self._is_collision(pt_r)) or
            (dir_d and self._is_collision(pt_l)) or
            (dir_l and self._is_collision(pt_u)) or
            (dir_r and self._is_collision(pt_d)),

            # Danger left
            (dir_d and self._is_collision(pt_r)) or
            (dir_u and self._is_collision(pt_l)) or
            (dir_r and self._is_collision(pt_u)) or
            (dir_l and self._is_collision(pt_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < head.x,   # food left
            self.food.x > head.x,   # food right
            self.food.y < head.y,   # food up
            self.food.y > head.y,   # food down
        ]
        return [int(s) for s in state]

    # ── Movement ───────────────────────────────────────────────────────────────
    def _move(self, action):
        # action: [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx        = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK
        elif self.direction == Direction.LEFT:  x -= BLOCK
        elif self.direction == Direction.DOWN:   y += BLOCK
        elif self.direction == Direction.UP:     y -= BLOCK
        self.head = Point(x, y)

    # ── Rendering ──────────────────────────────────────────────────────────────
    def _draw(self):
        self.display.fill(GRAY)

        for i, pt in enumerate(self.snake):
            color = GREEN1 if i == 0 else GREEN2
            pygame.draw.rect(self.display, color,
                             pygame.Rect(pt.x, pt.y, BLOCK, BLOCK))
            pygame.draw.rect(self.display, BLACK,
                             pygame.Rect(pt.x + 2, pt.y + 2, BLOCK - 4, BLOCK - 4))

        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK, BLOCK))

        txt = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(txt, (4, 4))
        pygame.display.flip()
        self.clock.tick(SPEED)
