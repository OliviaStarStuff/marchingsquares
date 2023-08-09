# Marching Squares test
![Preview of app](preview.png)
An implementation of binary marching squares
## Setup
### Requirements
* Pygame 2.4.0
* Python 3.9.12
### Installation
Clone this repository and run in folder using `python app.py` or `python3 app.py`.

## Controls
### Movement Controls
* W/S - Shrink Expand Y Axis
* A/D - Shrink/Expand on the X Axis
* Q/E - Zoom in/out
* Z/C - More/Less Details
### Toggles
* F - Flat shade dots
  * forgoes gradient and paints dots either black or white
* G - Show exterior dots
  * dots outside of boundary (black on gray)
* B - Show interior dots
  * dots inside of boundary (gray to white)
* Y - Shade exterior
* H - Shade interior
  * Shade the inside of the boundary
* N - Draw boundaries
* R - Reload
  * Regenerate map
* X - Cycle debug Mode (1=Group Value, 2=Cell Value, 4=Colours)
  * Uses 3 bitflags and cycles through 8
* T - Animate
## Todo
* Use simplex or perlin noise to generate maps
* Add relative boundary shaping
* Consider moving to 3D increased complexity 2<sup>8</sup> combinations.