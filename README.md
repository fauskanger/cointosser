# Trippel Coin-tosser

Given a start prediction of a coin flip sequence of length 3, a second prediction with the following rule will dominate:

 - Start prediction: (A, B, C)
 - Second prediction: (opposite of B, A , B)
 
Python-script [here](run.py)

PyPlot adapted from http://matplotlib.org/examples/pylab_examples/bar_stacked.html

UPDATE: It's been brought to my attention that this is called [Penney's Game][1].

Results for experiments with n = 1 000 000:

![Count](https://dl.dropboxusercontent.com/u/2563770/cointosscount1m.png)
![Ratio](https://dl.dropboxusercontent.com/u/2563770/cointosspercent1m.png)

[1]: https://en.wikipedia.org/wiki/Penney%27s_game

