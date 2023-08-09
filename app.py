import pygame as pg
import random
from edges import BOUNDARIES, Edge


WHITE = pg.Color(255, 255, 255)
BLACK = pg.Color(0, 0, 0)
ROTATION = pg.Vector2(-1, 1)

"""
The container for marching cube simulation app
"""
class App:
    running = True
    is_show_outer_dots = False
    is_show_inner_dots = True
    is_flat_dots = True
    is_inner_shaded = True
    is_debug_on = False
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
            if event.type == pg.KEYDOWN:
                self.check_user_input()

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
        instruction_text = "f:flat shade dots|g:paint exterior dots|"
        instruction_text += "v:paint interior dots|h:shade interior|"
        instruction_text +="r:reload|x:show debug"
        text = self.font.render(instruction_text, True, WHITE)
        self.screen.blit(text, self.CENTER + (-text.get_width()/2, self.CENTER.y - 24))
        text = self.font.render("WASD:Expand/Shrink|QE:Zoom in/out ", True, WHITE)
        self.screen.blit(text, self.CENTER + (-text.get_width()/2, -self.CENTER.y + 12))
        if self.is_debug_on:
            self.draw_debug_text()
        # maintain framerate
        self.dt = self.clock.tick(self.framerate)
        pg.display.flip()

    """Check for user input"""
    def check_user_input(self):
        keys = pg.key.get_pressed()
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
            self.is_debug_on = not self.is_debug_on
        if keys[pg.K_v]:
            self.is_show_inner_dots = not self.is_show_inner_dots
        # Restart
        if keys[pg.K_r]:
           self.restart()
        # Expand/Shrink/Scale
        if keys[pg.K_a]:
            self.col_num -= 1
            for i in range(self.row_num):
                self.grid[i] = self.grid[i][:-1]
        if keys[pg.K_d]:
            self.col_num += 1
            for i in range(self.row_num):
                self.grid[i].append(random.random())
        if keys[pg.K_w]:
            self.row_num -= 1
            self.grid = self.grid[:-1]
        if keys[pg.K_s]:
            self.row_num += 1
            self.grid.append([random.random() for i in range(self.col_num)])
        if keys[pg.K_q]:
            self.size = max(self.size-5, 0)
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
            origin = boundaries[i+1].copy()
            polygon = [boundaries[i], boundaries[i+1]]
            if boundary_id in [Edge.AB, Edge.CD, Edge.BD, Edge.AC]: #horizontal
                diff = (boundaries[i+1] - boundaries[i])/2
                origin = boundaries[i] - diff.yx
                origin_2 = boundaries[i+1] - diff.yx
                polygon.append(origin_2)
                polygon.append(origin)
            elif boundary_id in [Edge.ABC, Edge.ABD, Edge.ACD, Edge.BCD]: #horizontal
                if self.is_debug_on:
                    colour = (255,0,255)
                origin_3 = boundaries[i+1] * 2 - boundaries[i]
                origin_3.update([pg.math.clamp(p, 0 ,1) for p in origin_3])
                origin_2 = origin_3 - (0.5, 0.5)
                origin_2 = origin_2.yx.elementwise() * ROTATION
                origin = origin_2.yx.elementwise() * ROTATION

                polygon.append(origin_3)
                polygon.append(origin_2 + (0.5, 0.5))
                polygon.append(origin + (0.5, 0.5))
            elif boundary_id == Edge.ABCD:
                polygon = boundaries[::2]
            else:
                if origin.y == 1 and origin.x == 0.5:
                    origin.x = 1
                elif origin.y == 0 and origin.x == 0.5:
                    origin.x = 0
                elif origin.x == 1 and origin.y == 0.5:
                    origin.y = 0
                elif origin.x == 0 and origin.y == 0.5:
                    origin.y = 1
                polygon.append(origin)
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
    def restart(self):
        self.grid = self.randomGrid(
                self.row_num, self.col_num)

    def draw_debug_text(self):
        for row in range(self.row_num):
            for col in range(self.col_num):
                boundary_id = self.determine_boundary_value(row, col)
                coords = self.cell_coords_to_window_coords(row, col)
                text = self.font.render(f"{boundary_id}", True, (0,0,255))
                self.screen.blit(text, coords + (-6, -6))


def main():
    sim_specs = {"threshold":0.5, "framerate":40}
    grid_specs = {"size":20, "cols":30, "rows":30, "bg_colour": (64, 64, 80)}
    app = App((1600, 768), sim_specs, grid_specs)

    while app.running:
        app.update()

if __name__ == "__main__":
    main()
