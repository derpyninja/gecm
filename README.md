**G**ames in support of **E**cosystem **C**risis **M**anagement (gecm)
======================================================================

Welcome to our student project on **G**ames in support of **E**cosystem **C**risis **M**anagement. 
This repository was created as part of [Foundations of Ecosystem Management](https://ecology.ethz.ch/education/master-courses/foundations-of-ecosystem-management.html), 
a graduate-level course offered at **ETH Zurich** in the autumn semester of 2020. 
The team members are Martina, Marco, Lena, Ella-Mona and Felix.


Description
===========

If you want to co-develop the package, please follow the steps outlined [here](https://pypi.org/project/PyScaffold). 
In addition, [this blogpost](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/)
is an excellent entry point for learning how to organise your code well, especially
in view of using *jupyter notebooks* within *jupyter lab*.

The main bulk of game development will probably be done using *numpy* and *tkinter* following [this blogpost](https://towardsdatascience.com/making-simple-games-in-python-f35f3ae6f31a).
In addition, we could also try to leverage the following packages:

- [game2dboard](https://pypi.org/project/game2dboard)
- [pyGameMath](https://github.com/AlexMarinescu/pyGameMath)

Structure
=========

**Code Structure:**

To have a better and logical organization of the code, we will probably structure the functions structured into three categories (category 2 will not relate to tkinter in the end) based on [this blogpost](https://towardsdatascience.com/making-simple-games-in-python-f35f3ae6f31a).

1) **Initialization** functions: These functions are responsible for setting up an initial state for the game. These include defining the game variables, initializing the game graphics, resetting the game variables when the game is over, defining the canvas and bind widgets, etc. Initialization functions will mainly deal with setting up the game in case a new game is started or the game is concluded and needs to be played again.
2) **Drawing** functions: As the name suggests, these functions will be responsible to draw game-based graphical elements onto the Tkinter window. Based on the basic canvas drawing methods mentioned above, we will be creating high-level drawing functions specific to our game. These high-level drawing functions will then be used as building blocks for updating game graphics.
3) **Logical** functions: These functions will have nothing to do with the game graphics and will deal with the game logic. These include, but are not limited to, keeping track of the game state, receiving user input, updating the state of the game, checking if the current move is a legal one, keeping track of player scores, checking if the game has concluded, deciding on the result of the game, etc.

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
