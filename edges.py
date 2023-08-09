import pygame as pg
from enum import IntFlag, auto

"""
A,B,C, and D is used to refer each corner of a square starting with the
top right and going clockwise until the bottom right. arrange as so
A B
C D
"""
class Edge(IntFlag):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    AB = A | B
    AC = A | C
    AD = A | D
    BC = B | C
    BD = B | D
    CD = C | D
    ABC = AB | C
    ABD = AB | D
    ACD = AC | D
    BCD = BC | D
    ABCD = AB | CD

MID_AB = pg.Vector2(0.5, 0.0)
MID_BD = pg.Vector2(1.0, 0.5)
MID_AC = pg.Vector2(0.0, 0.5)
MID_CD = pg.Vector2(0.5, 1.0)

"""
pairs of x,y start and end coordinates for each line that needs to be drawn
depending on configuration of 2x2 cells. coordinates are in a 1x1 unit square
and contains values 1 - 15
"""
BOUNDARIES: dict[Edge, tuple[pg.Vector2, ...]] = {
    Edge.A   : (MID_AC, MID_AB), #1
    Edge.B   : (MID_AB, MID_BD), #2
    Edge.AB  : (MID_AC, MID_BD), #3
    Edge.C   : (MID_CD, MID_AC), #4
    Edge.AC  : (MID_AB, MID_CD), #5
    Edge.BC  : (MID_AB, MID_BD,
                MID_CD, MID_AC), #6
    Edge.ABC : (MID_BD, MID_CD), #7
    Edge.D   : (MID_BD, MID_CD), #8
    Edge.AD  : (MID_AC, MID_AB,
                MID_BD, MID_CD), #9
    Edge.BD  : (MID_CD, MID_AB), #10
    Edge.ABD : (MID_CD, MID_AC), #11
    Edge.CD  : (MID_BD, MID_AC), #12
    Edge.ACD : (MID_AB, MID_BD), #13
    Edge.BCD : (MID_AC, MID_AB), #14
    Edge.ABCD: (pg.Vector2(0, 0), pg.Vector2(1, 0),
                pg.Vector2(1, 0), pg.Vector2(1, 1),
                pg.Vector2(1, 1), pg.Vector2(0, 1),
                pg.Vector2(0, 1), pg.Vector2(0, 0))  #15
}
