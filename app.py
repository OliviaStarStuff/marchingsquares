import pygame as pg
import random
from edges import BOUNDARIES, Edge


WHITE = pg.Color(255, 255, 255)
BLACK = pg.Color(0, 0, 0)
ROTATION = pg.Vector2(-1, 1)
INSTRUCTION_TEXT = "f:flat shade dots|g:paint exterior dots|"\
                 + "v:paint interior dots|h:shade interior|"\
                 + "r:reload|x:show debug"
"""
The container for marching cube simulation app
"""
class App:
    running = True
    is_show_outer_dots = False
    is_show_inner_dots = True
    is_flat_dots = True
    is_inner_shaded = True
    is_debug_on = 0
    interval = 200
    last_time = 0
    dt = 0

    def __init__(self, dimensions: tuple[int, int], sim_values: dict[str, any],
                 grid_values: dict[str, any]) -> None:
        # Setup Pygame
        pg.init()
        self.screen = pg.display.set_mode(dimensions)
        self.CENTER = pg.Vector2(
            self.screen.get_width(), self.screen.get_height()) / 2
        self.font = pg.font.SysFont('Consolas', 18)
        self.clock = pg.time.Clock()

        # Setup grid settings
        self.size: int = grid_values.get("size")
        self.row_num: int = grid_values.get("rows")
        self.col_num: int = grid_values.get("cols")
        self.grid = self.randomGrid(self.row_num, self.col_num)
        self.background_colour = grid_values.get("bg_colour")

        # Setup simulation settings
        self.threshold = sim_values.get("threshold")
        self.framerate = sim_values.get("framerate")

    """Update loop for the app"""
    def update(self):
        # poll for events
        # pg.QUIT event means the user clicked X to close your window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            self.check_user_input(event)

        # fill the screen with a color to wipe away anything from last frame
        self.screen.fill((64,64,80))

        # iterate through each cell coordinates,
        # determine window coords and draw dots and boundary lines
        for row in range(self.row_num):
            for col in range(self.col_num):
                coords = self.cell_coords_to_window_coords(row, col)
                boundary_id = self.determine_boundary_value(row, col)
                if self.is_inner_shaded:
                    self.fill_boundary_spaces(boundary_id, coords)
                self.draw_dots(row, col, coords)
                self.draw_boundary_lines(boundary_id, coords)

        text = self.font.render(INSTRUCTION_TEXT, True, WHITE)
        self.screen.blit(text, self.CENTER + (-text.get_width()/2, self.CENTER.y - 24))
        text = self.font.render("WASD:Expand/Shrink|QE:Zoom in/out ", True, WHITE)
        self.screen.blit(text, self.CENTER + (-text.get_width()/2, -self.CENTER.y + 12))
        if self.is_debug_on > 0:
            self.draw_debug_text()
        # maintain framerate
        self.dt = self.clock.tick(self.framerate)
        pg.display.flip()

    """Check for user input"""
    def check_user_input(self, event: pg.event) -> None:
        keys = pg.key.get_pressed()
        if event.type == pg.KEYDOWN:
        # Toggles
            if keys[pg.K_f]:
                self.last_time = pg.time.get_ticks()
                self.is_flat_dots = not self.is_flat_dots
            if keys[pg.K_g]:
                self.last_time = pg.time.get_ticks()
                self.is_show_outer_dots = not self.is_show_outer_dots
            if keys[pg.K_h]:
                self.last_time = pg.time.get_ticks()
                self.is_inner_shaded = not self.is_inner_shaded
            if keys[pg.K_x]:
                self.is_debug_on = (self.is_debug_on + 1) % 8
            if keys[pg.K_v]:
                self.is_show_inner_dots = not self.is_show_inner_dots
            # Restart
            if keys[pg.K_r]:
                self.restart()
        # Expand/Shrink/Scale
        if keys[pg.K_a]:
            temp = self.col_num-1
            self.col_num = max(self.col_num-1, 1)
            if temp == self.col_num:
                for i in range(self.row_num):
                    self.grid[i] = self.grid[i][:-1]
        if keys[pg.K_d]:
            self.col_num = min(self.col_num+1, 100)
            for i in range(self.row_num):
                self.grid[i].append(random.random())
        if keys[pg.K_w]:
            temp = self.row_num-1
            self.row_num = max(self.row_num-1, 1)
            if temp == self.row_num:
                self.grid = self.grid[:-1]
        if keys[pg.K_s]:
            self.row_num = min(self.row_num+1, 100)
            self.grid.append([random.random() for i in range(self.col_num)])
        if keys[pg.K_q]:
            changed = True
            self.size = max(self.size-5, 5)
        if keys[pg.K_e]:
            self.size = min(self.size+5, 100)


    """Draws the boundary lines"""
    def draw_boundary_lines(self, boundary_id: int, coords: pg.Vector2) -> None:
        colour = WHITE
        if boundary_id == Edge.ABCD:
            return
        boundaries = BOUNDARIES.get(boundary_id, ())
        for i in range(0, len(boundaries), 2):
            start_pos = boundaries[i]*self.size + coords
            end_pos = boundaries[i+1]*self.size + coords
            pg.draw.line(self.screen, colour, start_pos, end_pos)

    """Fills bounded space"""
    def fill_boundary_spaces(self, boundary_id: int, coords: pg.Vector2) -> None:
        boundaries = BOUNDARIES.get(boundary_id, ())
        colour = (0,0,255)
        for i in range(0,len(boundaries), 2):
            center_point = boundaries[i+1].copy()
            polygon = [boundaries[i], boundaries[i+1]]
            if boundary_id in [Edge.AB, Edge.CD, Edge.BD, Edge.AC]: #horizontal
                if self.is_debug_on & 4:
                    colour = (0, 224, 64)
                diff = (boundaries[i+1] - boundaries[i])/2
                left_point = boundaries[i] - diff.yx
                right_point = boundaries[i+1] - diff.yx
                polygon.append(right_point)
                polygon.append(left_point)
            elif boundary_id in [Edge.ABC, Edge.ABD, Edge.ACD, Edge.BCD]: #horizontal
                if self.is_debug_on & 4:
                    if boundary_id == Edge.ABC:
                        colour = (255, 0, 255)
                    if boundary_id == Edge.ABD:
                        colour = (128, 0, 80)
                    if boundary_id == Edge.ACD:
                        colour = (200, 0, 100)
                    if boundary_id == Edge.BCD:
                        colour = (144, 0, 144)
                right_point = boundaries[i+1] * 2 - boundaries[i]
                right_point.update([pg.math.clamp(p, 0 ,1) for p in right_point])
                center_point = right_point - (0.5, 0.5)
                center_point = center_point.yx.elementwise() * ROTATION
                left_point = center_point.yx.elementwise() * ROTATION

                polygon.append(right_point)
                polygon.append(center_point + (0.5, 0.5))
                polygon.append(left_point + (0.5, 0.5))
            elif boundary_id == Edge.ABCD:
                if self.is_debug_on & 4:
                    colour = (224, 224, 64)
                polygon = boundaries[::2]
            else:
                if self.is_debug_on & 4:
                    colour = (0, 48, 80)
                    if boundary_id in [Edge.A, Edge.B, Edge.C, Edge.D]:
                        colour = (0, 128, 180)
                if center_point.x == 0.5:
                    center_point.x = 1 if center_point.y == 1 else 0
                elif center_point.y == 0.5:
                    center_point.y = 0 if center_point.x == 1 else 1
                polygon.append(center_point)
            polygon_2 = [point * self.size + coords for point in polygon]
            pg.draw.polygon(self.screen, colour, polygon_2)

    """Draws all dots"""
    def draw_dots(self, row, col, coords):
        colour = self.determine_cell_colour(row, col)
        if self.is_show_outer_dots and self.grid[row][col] < self.threshold:
            pg.draw.circle(self.screen, colour, coords, self.size/4)
        if self.is_show_inner_dots and self.grid[row][col] >= self.threshold:
            pg.draw.circle(self.screen, colour, coords, self.size/4)

    """Transates cell coordinates to window coords"""
    def cell_coords_to_window_coords(self, row: int, col: int) -> pg.Vector2:
        x = col*self.size + self.CENTER.x - self.col_num*self.size / 2
        y = row*self.size + self.CENTER.y - self.row_num*self.size / 2
        return pg.Vector2(x, y)

    """
    Determines the boundary type from adjacent cells
    used to query BOUNDARIES
    """
    def determine_boundary_value(self, row: int, col: int) -> Edge:
        boundaries = 0
        for i in range(2):
            row_index = (row+i) % self.row_num
            for j in range(2):
                modifier = 2 ** (i * 2 + j)
                col_index = (col+j) % self.col_num
                cell = self.grid[row_index][col_index]
                boundaries += (cell > self.threshold) * modifier
        return boundaries

    """variable boundaries, unused WIP and incomplete"""
    def select_variable_boundary(self, row: int, col: int) -> int:
        get_row = 2 if row != len(self.grid)-1 else 1
        get_col = 2 if col != len(self.grid[row])-1 else 1
        boundaries = [[0, 0], [0, 0]]
        for i in range(get_row):
            for j in range(get_col):
                boundaries[i][j] = self.grid[row+i][col+j]
        # set positions for a b c d as needed
        a = boundaries[0][0]/boundaries[0][0] + boundaries[0][1]
        b = boundaries[0][1]/boundaries[0][1] + boundaries[1][1]
        c = boundaries[1][0]/boundaries[1][0] + boundaries[1][1]
        d = boundaries[1][0]/boundaries[1][0] + boundaries[1][1]

    """Sets colour based on whatever algorithm with use"""
    def determine_cell_colour(self, row: int, col: int) -> pg.Color:
        if self.is_flat_dots:
            return WHITE if self.grid[row][col] > self.threshold else BLACK

        return BLACK.lerp(WHITE, self.grid[row][col])

    """Random function used to generate grid"""
    def randomGrid(self, rows: int, columns: int) -> list[list[float]]:
        return [[random.random() for i in range(columns)] for j in range(rows)]

    """restart the simulation"""
    def restart(self) -> None:
        self.grid = self.randomGrid(
                self.row_num, self.col_num)

    def draw_debug_text(self) -> None:
        for row in range(self.row_num):
            for col in range(self.col_num):
                text_height = -8
                coords = self.cell_coords_to_window_coords(row, col)
                if self.is_debug_on & 1:
                    boundary_id = self.determine_boundary_value(row, col)
                    text_content = f"{boundary_id}"
                    text = self.font.render(text_content, True, WHITE)
                    self.screen.blit(text, coords + (-12, text_height))
                if self.is_debug_on & 2:
                    text_content = f"{self.grid[row][col]:.1f}"
                    text = self.font.render(text_content, True, WHITE)
                    if self.is_debug_on & 1:
                        text_height += 14
                    self.screen.blit(text, coords + (-text.get_width()/2, text_height))


def main() -> None:
    sim_specs = {"threshold":0.5, "framerate":40}
    grid_specs = {"size":20, "cols":30, "rows":30, "bg_colour": (64, 64, 80)}
    app = App((1600, 768), sim_specs, grid_specs)

    while app.running:
        app.update()

if __name__ == "__main__":
    main()
