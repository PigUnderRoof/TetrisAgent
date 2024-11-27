import numpy as np
from numba import njit
from typing import Tuple, List, Dict, Any
import pygame

Array = np.ndarray

GridWidth: int = 10
GridHeight: int = 20
BufferHeight: int = 4


Tetrominoes = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.int8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.int8),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
}


class Action:
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROTATE = 3
    NoOps = 4


CellSize = 30
CellPadding = 1
HudHeight = CellSize
Black = (0, 0, 0)
White = (255, 255, 255)
ScoreColor = (0, 0, 255)


@njit
def rotate(arr, n):
    return np.rot90(arr, n)


class TetrisGame:
    def __init__(self, w: int = GridWidth, h: int = GridHeight, render_delay: float = 0):
        # layout
        self.W: int = w
        self.H: int = h

        self.reset()

        # ui
        self.render_delay: float = render_delay
        self.render_game: bool = render_delay > 0
        if self.render_game:
            self._setup_screen()

    def reset(self):
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)

        self.piece = None
        self.piece_pos = np.zeros(2, dtype=np.int8)
        self.piece_shape: str = ""

        self.step_count = 0
        self.score = 0
        self.done = False

        self._spawn_piece()
        return self._get_observation()

    def is_valid_position(self, piece: Array, pos: Array) -> bool:
        h, w = piece.shape
        x, y = pos

        # check out of bound
        if x < 0 or x + w > self.W or y + h > self.H + BufferHeight:
            return False

        for y_ in range(h):
            for x_ in range(w):
                if y + y_ - BufferHeight < 0:
                    continue

                if piece[y_, x_] == 0:
                    continue

                if self.grid[y + y_ - BufferHeight, x + x_] != 0:
                    return False
        return True

    def step(self, a: Action) -> Tuple[Array, float, bool, Dict[str, Any]]:
        reward = 0.
        done = False

        if self.piece is None:
            self._spawn_piece()

        tmp_piece = self.piece.copy()
        tmp_pos = self.piece_pos.copy()
        if a == Action.LEFT:
            tmp_pos[0] -= 1

        elif a == Action.RIGHT:
            tmp_pos[0] += 1
        elif a == Action.DOWN or a == Action.NoOps:
            tmp_pos[1] += 1

        if a == Action.ROTATE:
            tmp_piece = rotate(self.piece, 3)

        # move piece
        if self.is_valid_position(tmp_piece, tmp_pos):
            self.piece = tmp_piece.copy()
            self.piece_pos = tmp_pos.copy()
        else:
            # check lock
            tmp_pos[1] += 1
            if action == Action.DOWN and (not self.is_valid_position(self.piece, tmp_pos)):
                self._lock_piece()

                # clear lines
                num_lines = self.clear_lines()
                reward += num_lines
                self.score += int(reward * 100)

                done = self.check_done()

                # spawn new piece
                if not done:
                    self._spawn_piece()
                else:
                    reward = -1.

        self.step_count += 1
        info = {
            'grid': self.grid.copy(),
            'piece': self.piece.copy(),
            'piece_pos': self.piece_pos.copy(),
            'piece_shape': self.piece_shape,
            'score': self.score,
            'step': self.step_count,

        }

        if self.render_game:
            self.render()

        obs = self._get_observation()
        return obs, reward, done, info

    def _lock_piece(self):
        h, w = self.piece.shape
        x, y = self.piece_pos

        for y_ in range(h):
            for x_ in range(w):
                if y + y_ - BufferHeight < 0:
                    continue

                if self.piece[y_, x_] == 0:
                    continue

                self.grid[y + y_ - BufferHeight, x + x_] = self.piece[y_, x_]

    def list_lines_to_clear(self) -> List[int]:
        rows_to_clear = []

        for y in range(self.H):
            row_idx = self.H - 1 - y
            if sum(self.grid[row_idx]) == self.W:
                rows_to_clear.append(row_idx)
        return rows_to_clear

    def clear_lines(self) -> int:
        rows_to_clear = self.list_lines_to_clear()

        if len(rows_to_clear) > 0:
            grid = np.zeros((self.H, self.W), dtype=np.int8)

            for y in range(self.H):
                row = self.grid[y]
                if y in rows_to_clear:
                    continue

                if sum(row) == 0:
                    continue

                drop_height = 0
                for row_idx in rows_to_clear:
                    if y < row_idx:
                        drop_height += 1

                grid[y + drop_height] = row.copy()
            self.grid = grid

        return len(rows_to_clear)

    def check_done(self) -> bool:
        max_height = self.piece_pos[1] + self.piece.shape[0]
        if max_height < BufferHeight:
            return False
        return self.piece_pos[1] < BufferHeight

    def render(self):
        if self.screen is None:
            return  # 如果不需要渲染，直接返回

        self.screen.fill(Black)  # 清屏

        obs = self._get_observation()
        h, w = obs.shape

        for y in range(h):
            for x in range(w):
                rect = (
                    x * CellSize,
                    y * CellSize + HudHeight,
                    CellSize - CellPadding,
                    CellSize - CellPadding,
                )

                pygame.draw.rect(
                    self.screen,
                    White,
                    rect,
                    int(obs[y, x] == 0)
                )

         # Draw buffer danger line
        pygame.draw.line(
            self.screen,
            ScoreColor,
            (0, BufferHeight * CellSize + HudHeight - 1),
            (self.W * CellSize, BufferHeight * CellSize + HudHeight - 1),
            2,
        )

        # 显示分数
        score_surface = self.font.render(f"Score: {self.score:d}", True, White)
        self.screen.blit(score_surface, (2, 2))

        pygame.display.flip()
        self.clock.tick(60)

    def _setup_screen(self):
        pygame.init()
        self.ui_w = self.W * CellSize
        self.ui_h = (self.H + BufferHeight) * CellSize + HudHeight
        self.screen = pygame.display.set_mode((self.ui_w, self.ui_h))

        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

    def close(self):
        pygame.quit()

    def _get_observation(self):
        buffer = np.zeros((BufferHeight, self.W), dtype=np.int8)
        grid = self.grid.copy()

        array = np.concatenate([buffer, grid], axis=0)

        if self.piece is not None:
            offset_x, offset_y = self.piece_pos
            h, w = self.piece.shape
            for y in range(h):
                for x in range(w):
                    if self.piece[y, x] != 0:
                        array[y + offset_y, x + offset_x] = self.piece[y, x]

        return array

    def _spawn_piece(self):
        idx = np.random.randint(len(Tetrominoes))
        shape = list(Tetrominoes.keys())[idx]

        rotation_idx = np.random.randint(4)
        piece = rotate(Tetrominoes[shape], rotation_idx)
        self.piece = piece

        x = self.W // 2 - piece.shape[1] // 2
        self.piece_pos = np.array([x, 0], dtype=np.int8)


if __name__ == "__main__":
    import sys
    import time

    env = TetrisGame(render_delay=0.2)
    obs = env.reset()
    done = False
    paused = False  # 用于控制暂停
    restart = False  # 用于控制重新开始

    action = None  # 当前的动作

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0  # 左移
                elif event.key == pygame.K_RIGHT:
                    action = 1  # 右移
                elif event.key == pygame.K_DOWN:
                    action = 2  # 旋转
                elif event.key == pygame.K_UP:
                    action = 3  # 软降
                elif event.key == pygame.K_SPACE:
                    action = 4  # 无运动
                elif event.key == pygame.K_p:
                    paused = not paused  # 切换暂停状态
                elif event.key == pygame.K_r:
                    restart = True  # 重新开始
            elif event.type == pygame.KEYUP:
                action = None  # 释放按键时不执行动作

        if restart:
            obs = env.reset()
            done = False
            paused = False
            restart = False
            action = None
            continue

        if not paused and not done:
            if action is not None:
                obs, reward, done, info = env.step(action)
                action = None  # 每次只处理一次动作
            env.render()
            time.sleep(env.render_delay)
        elif paused:
            # 暂停时显示暂停画面
            # env.render()
            env.screen.fill(Black)
            pause_surface = env.font.render("Paused", True, (255, 0, 0))
            env.screen.blit(
                pause_surface, (env.ui_w // 2 - 50, env.ui_h // 2))
            pygame.display.flip()
        else:
            # 游戏结束，显示 Game Over
            env.screen.fill(Black)
            game_over_surface = env.font.render("Game Over", True, (255, 0, 0))
            restart_surface = env.font.render(
                "Press R to Restart", True, (255, 255, 255)
            )
            env.screen.blit(
                game_over_surface, (env.ui_w // 2 - 70, env.ui_h // 2 - 30)
            )
            env.screen.blit(
                restart_surface, (env.ui_w // 2 - 100, env.ui_h // 2 + 10)
            )
            pygame.display.flip()

    env.close()
