# Handwritten Digits Image Classification

[MNIST](http://yann.lecun.com/exdb/mnist/) is a popular data set containing simple images of handwritten digits 0-9. 
Every digit is represented by a 28 x 28 pixel image. 
There is a long-term hosted competition on [Kaggle](https://www.kaggle.com/c/digit-recognizer) using MNIST.
This tutorial shows how to use MXNet in R to develop neural network models for competing in this multi-class classification challenge.


## Loading the Data

First, let's download the data from [Kaggle](http://www.kaggle.com/c/digit-recognizer/data) and put it in a ``data/`` sub-folder inside the current working directory.  

```{.python .input  n=1}
if (!dir.exists('data')) {
    dir.create('data')
}
if (!file.exists('data/train.csv')) {
    download.file(url='https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/train.csv',
                  destfile='data/train.csv', method='wget')
}
if (!file.exists('data/test.csv')) {
    download.file(url='https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/test.csv',
                  destfile='data/test.csv', method='wget')
}
```

The above commands rely on ``wget`` being installed on your machine.
If they fail, you can instead manually get the data yourself via the following steps:

1) Create a folder named ``data/`` in the current working directory (to see which directory this is, enter: ``getwd()`` in your current R console). 

2) Navigate to the [Kaggle website](https://www.kaggle.com/c/digit-recognizer/data), log into (or create) your Kaggle account and accept the terms for this competition.

3) Finally, click on **Download All** to download the data to your computer, and copy the files ``train.csv`` and ``test.csv`` into the previously-created ``data/`` folder.

Once the downloads have completed, we can read the data into R and convert it to matrices:

```{.python .input  n=2}
require(mxnet)
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test_features <- data.matrix(test) # Labels are not provided for test data.
train_features <- train[,-1]
train_y <- train[,1]
train_x <- t(train_features/255)
test_x <- t(test_features/255)
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Loading required package: mxnet\n"
 }
]
```

Every image is represented as a single row in ``train.features``/``test.features``. The greyscale of each image falls in the range [0, 255]. Above, we linearly transform the pixel values into [0,1]. We have also transposed the input matrix to *npixel* x *nexamples*, which is the major format for columns accepted by MXNet (and the convention of R).

Let's view an example image:

```{.python .input  n=3}
i = 10 # change this value to view different examples from the training data

pixels = matrix(train_x[,i],nrow=28, byrow=TRUE)
image(t(apply(pixels, 2, rev)) , col=gray((0:255)/255), 
      xlab="", ylab="", main=paste("Label for this image:", train_y[i]))
```

```{.json .output n=3}
[
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0gAAANICAIAAAByhViMAAAABmJLR0QA/wD/AP+gvaeTAAAg\nAElEQVR4nOzde3yU5Z3w/2sMEUKQgwHkFPEQBS0quKZWsUpErW052NKiZQWp+qhtrFptobUY\nBduubvdBSxWRR2xVVrfqlopdJK4aVkAOLRW1jcQTUaxEDQFEIZzM74/5bV55hYMxDJnMxfv9\nR1/NzDXXfMkdyMc7c08SdXV1AQCAzHdQugcAACA1hB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBA\nJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0A\nQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQd\nAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSE\nHQBAJIQdxOCNN95IJBKJRGLBggWp2vOqq65KJBK33HJLqjYMIcyePTuRSJxyyil7X/bRRx9d\nfvnlPXv2zMrKmjFjRgoHaIq77rorkUicc845TVx5xhlnpOQZ930fAGEHtDo/+9nPZs2aVVVV\n9emnn6Z7FvaXurq63/72t0VFRd26dcvOzs7LyzvrrLPuvvvu7du3p3s0yGDCDmh1Fi5cGEK4\n9tprt2zZctVVV+3X52rTpk0ikVi5cmUzHnv11VfX1dUtWrRoH2dI1T4ZpK6ubvTo0ZdeeumC\nBQuqq6t37NhRU1Pz/PPPX3311WeeeebHH3+c7gEhUwk7oNXZsGFDCKGoqKhdu3bpnoX94ne/\n+93jjz/epk2bX/ziF6tXr966detbb701YcKEEMLSpUtvv/32dA8ImUrYAa1U+/bt0z0C+8t9\n990XQvjJT35y4403HnHEEQcffPCRRx55++23X3HFFSGEJ598Mt0DQqYSdnAgWrBgwUUXXXT0\n0Ufn5OT07t37zDPPvP/++7du3brbxVVVVVdddVXv3r3btWtXUFDwwx/+8IMPPth12fr16ydN\nmnTCCSfk5uZ27tz59NNPnzZt2p723JNvfetbiUTi7bffDiGcd955iUSi4cUTb7755pVXXnnU\nUUe1a9euS5cuZ5999oMPPtjodXjJ6zMuuOCCLVu2TJo0qW/fvkceeeRun+v8889PJBI7d+4M\nIQwaNCiRSDT6Yej27dtvu+22448/vn379l27dh0+fPiKFSsaLtj1ooeNGzfeeOONxx9/fE5O\nTm5u7he+8IWf/vSn69ev3/ufutE+yQ+HDRu2Y8eOX/3qV/3792/fvn1BQcGtt96afP3ZAw88\ncMopp7Rv3z4vL2/EiBF///vfG23Y9OP7wgsvfP3rX+/SpUunTp3OOOOMOXPm7NixI5FI7Hqu\ntInHd/78+cnreJYuXbqXP3JFRUUIYdiwYY1uHzx4cAjho48+2vtnDNijOiDzvf7668m/0WVl\nZZ+5+Oabb97tvwYnn3zypk2b6pddeeWVIYTvfOc73bt3b7SyV69er776asM9X3755V69eu26\n57HHHvvOO+/UL3vooYdCCP/0T/+0p9lGjRrVaId77rknedecOXN2+5PZc8455+OPP270FF/9\n6le//OUvJxf07dt3t8/1la98pdFWCxcurKur+81vfhNCGDJkyNe+9rVGC9q2bfvXv/61fofk\nysGDByc//OCDDwoKCnadsKCg4P3339/LEWm0T/LDr3zlK7t2T3Fx8Q9+8INGN3bq1KnhJ7mJ\nx7euru7ee+896KDG/3lfUlKS/JM24/jW1dU99dRTybuWLFmylz/ynlx33XUhhK9+9avNeCxQ\nV1cn7CAGTQ+7tWvXZmVlhRB+8IMfvP7667W1tR9++OHvf//7rl27hhD+5V/+pX5lMuxCCD16\n9HjooYfWr1//8ccfP/bYY926dQshnHDCCZ9++mlyZU1NTX5+fgjh7LPPXrhw4aZNm2pqah5/\n/PG+ffuGEL70pS/Vr/zMsEtKPrDhn+WVV15JVt03v/nNl156qba2tqqq6p577unYsWMI4cIL\nL6xfmXyKgw46KCcn55577qmurt77cyU/Gy+++GL9LcmuSp61uvnmm998883a2tqysrI+ffqE\nEEaPHt1oZX2QXXPNNSGEE088cenSpVu2bNm0adMzzzzTv3//ZJDtZYbdht1BBx2Um5v761//\nuqqqas2aNePHj6/PqQsvvPDvf//7J5988swzz3Tp0iWE8LOf/Sz52KYf3+XLlydXfve73121\nalVtbW15eXl9WDcMu6Yf32bbvn37W2+9NWXKlKysrKysrGRhA80g7CAGTQ+7J554InnyptHt\nydM8F110Uf0tybBr165dRUVFw5WlpaXJ5yotLU3ectNNNyW7ZPv27Q1Xvvrqq8l0WLZsWfKW\nZofdN77xjRDCaaedtnPnzoYr586dmxzmlVdeafgUocGpvr3bU9iFEB588MGGK2fOnBlC6Nmz\nZ6OV9UE2aNCgEMLdd9/d8FH/8z//E0IoKCjYywy7DbsQwmOPPVa/ZuPGjcmza+ecc07Dx06c\nODGEcN555yU/bPrxTZ4O/OY3v9lw2aeffnreeec1CrumH9/mGTp0aH2z5ubmPvXUU/uyGxzg\nvMYODiwjRoyoq6tr9Fqx7du3J1+ntetbiI0aNerYY49teMt555130kknhRDq3wz5kUceCSF8\n73vfa9OmTcOV/fv3P+2000IIixcv3peZt2zZ8qc//SmEMGHChEY/Ohw+fPjRRx8dQpg/f37D\n23Nzc5Mvw2+2zp07//M//3PDW44//vgQwocffrinhyRPnt11110N3zzlzDPPrGtQ3k136KGH\nNvzBdMeOHTt37hxCuPDCCxsuS55HXLduXfLDJh7frVu3Pv300yGE5FnGeolE4vrrr280yf4+\nvg198sknl1566d5fnwfsRZvPXgJE54033nj22Wdfeumld9555+233169evUnn3yy25Unn3zy\nrjcOGjTopZdeWrNmTQhh06ZNb7zxRgjh4osvvvjii3e7yW4vtmi6v/3tb8kiOeuss3a994QT\nTnjzzTffeuuthjcWFBTs+uqxz6Vfv36NdsjNzQ0h7NixY8eOHY0SJ+mmm2564YUXXn311UGD\nBh111FFf/vKXv/zlL5977rmHH354MwY49thjE4lEw1uys7NDCLt9rduOHTsafviZx7eiomLb\ntm0hhC9+8YuNtvqnf/qnhh+2wPF95plnamtr165dW1paevvtt1dWVn7ta19bvXp1p06d9mVb\nODAJOziwbN269ZprrrnvvvsaXkxaWFjYu3fvP/7xj7uu3+17juTl5SW3CiHU1NQ05UmbP/H/\nPkVubm7ylNhuJ2z0FLsNr8+lQ4cOn/chQ4YMeemll/7t3/7tiSeeeOutt956660HHngghDB4\n8OBp06btNpH3ItmRu9p7sDbx+CYv1M3JycnJydn787bA8Q0htGvX7sgjj7zqqqvOP//8Y445\nZv369Y899tjll1++j9vCAciPYuHAMmHChJkzZ3766aejRo2aMWPGggULPvjgg+XLlzc6T1Nv\nt9/X//GPf4T/zbv68tvLy/umTp26LzMnL5vYvHlzbW3trve+++67IYTkxQFpd+yxx86cObOq\nqurll1++6667vvGNb+Tk5CxevPi8887buHFjCwzQxOPbtm3bEEJtbW3yvF1DjU6/7Y/j+7vf\n/S6RSOz21OMRRxzxhS98IYTw5ptvfq49gSRhBweWhx9+OIQwadKkxx9//MorrzzrrLOSV7lW\nVVXtdv3y5csb3VJXV/eXv/wlhDBw4MAQQteuXQ899NAQwksvvbSfZj7mmGOSz/vXv/610V2b\nNm1KDpO8cKGVSCQSJ5xwQnFx8R/+8IeKiorOnTuvW7fu+eefb4GnbuLxTUZVXV1d8sesDZWV\nlTX8cH8c3yOOOCKEsHbt2urq6l3vTbbmIYcckqqngwOKsIMDS/IMXGFhYcMb165d++ijj+52\n/fz58xu98H/u3LlvvPFGmzZtRo4cGUJIvpVuCOE3v/lNo2svqqqqDj300EQi8cILL+zLzL16\n9UpG5G233dborvvvv3/z5s25ubm7vudcC3vzzTcTiURubu7mzZsb3p6fn5+86KHRC+b2kyYe\n38MPPzz5DibTpk1rePu2bdv+7//9vw1v2R/H99RTT02+T82sWbMa3fX888+/+uqrYQ+vpwQ+\nk7CDA8uJJ54YQrj55ptXrFixdevWNWvW3HvvvV/84heTPyj88MMPk29pUb9+27Zt55577ty5\ncz/66KOPP/74P/7jPy699NIQwlVXXdWjR4/kmhtvvLF9+/ZvvvnmyJEjV65cuW3btvfee++R\nRx45/fTT169fX1RUdPrpp+/j2Ml3zX3yyScvu+yyVatWbd++/f333582bdpPfvKTEMKECROS\nodAMyZ/z/uUvf2n4p26Go48+ul+/fps3b/7Wt761bNmy5A+OX3755UsuuaSysrJjx471b5i8\nXzX9+F599dUhhJkzZ06ZMuWDDz7YsmXLCy+8cO6555aXlzfaM+XHNycnJ/nsJSUlt99++7vv\nvrt169Y333zztttuGz58eAjh7LPPTv4KCuBzS+FbpwDp0pR303j99dfr6urmz5+ffO+xhgYN\nGpR8/7OktWvX1v3v+9jdcMMNu16yMHjw4Ia/7KGuru7JJ5/c7WUWJ510UnK3pGa/j13dnn+h\nwqhRoxq+v1oTn6Jewyhp+Jsnhg4d2mjliy++mFxW/3SN3n/u2WefPfjgg3edMCsr65FHHtnL\nDLt9H7tdBzjssMNCCI3e5i25+KSTTkp+2PTju2PHjuS7AzaS/Dzn5OQ0fJYmHt+6Jv/miS1b\nthQVFe32gJ5yyikffvjhXh4L7IUzdnBg+cpXvvLss88WFRW1b9/+kEMOKSws/M1vfrNs2bIR\nI0b84Ac/yM3NPeywwxrWSf/+/ZcsWfLtb3+7W7du7dq1GzBgwG233fbcc881unZy2LBhL7/8\n8uWXX56fn5+dnd29e/czzjjj17/+9fLly+tP7O2jW2655dlnnx0xYkS3bt2ys7O7det23nnn\nPfzww4899ti+XAM7bdq0QYMG7TbIPq+zzz57+fLlF1988RFHHNG2bdvs7Ow+ffpcdNFFS5Ys\nueiii/Z9/6Zo+vHNysr6z//8z1mzZp166qk5OTmHHHLI6aef/sc//nH06NEhhEZXy6b8+LZr\n1660tPSuu+4aPHhw586d27Rp07Vr17PPPnvGjBmLFy9uJZfCQCZK1O3bTx8AiMmf/vSn4cOH\nn3TSSQ3fZhnIFM7YARyICgsLE4nEL37xi0a3P/jggyGEfX9ZJJAWwg7gQPT1r389hHD77bfP\nmjXr/fff37p1a3l5+ZVXXvnYY49lZ2d/73vfS/eAQHP4USzAgejjjz8+66yzdn1rwKysrLvu\nuuuqq65Ky1TAPhJ2AAeorVu3PvDAAw888MDrr7++adOmww477Mwzz7z66qt3/QWyQKYQdgAA\nkfAaOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7\nAIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgI\nOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASLRJ9wDN\ntHbt2kWLFq1evXrTpk25ubndu3cvKCgYPHhwVlZWukcDAEiPzAu7ysrK4uLiefPm7XpXXl7e\n2LFjJ0+e3LFjx5YfDAAgvRJ1dXXpnuFzWLNmzcknn1xdXd2hQ4ehQ4cOGDCga9euiURi/fr1\n5eXl8+fP37Rp03HHHbd48eIuXbqk6kk3btz4wAMPbNmyJVUbAgAZLScn55JLLunUqVO6B9lF\nXUYZP358CGHkyJGbNm3a9d7q6uovfelLIYTrrrsuhU969913p/kgAQCtzN13353C2EiVDLt4\norS0NIQwderUDh067HpvXl7ejBkzQghz5sxJ4ZNu3749hbsBABFonXmQYWFXU1MTQujdu/ee\n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\nhDB69OgWHgwAIO0y7DV2EydOXLhwYVlZWVFRUX5+/sCBA7t16xZCqKmpWblyZWVlZQihsLCw\npKQkzYMCALS4DAu79u3bl5aWzpw5c/r06eXl5WvWrGl4b58+fa644ooJEya0bds2XRMCAKRL\nhoVdCCE7O7u4uLi4uLiqqqqioqKmpmbbtm2dOnUqKCgoKChI93QAAGmTeWFXr0ePHj169Ej3\nFAAArUWGXTwBAMCeCDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7\nAIBICDsAgEgIOwCASAg7AIBICDsAgEi0SfcAAPtd27ZtU7LPtGnTUrJPCOHyyy9P1VZxe/XV\nV1O11de+9rWU7PPuu++mZB/YH5yxAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCI\nhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMA\niISwAwCIhLADAIiEsAMAiESbdA8A+9eRRx6Zqq2uvPLKlOzz3HPPpWSfEMKCBQtStdW2bdtS\nsk/v3r1Tsk8IoX///qnaauLEiSnZZ+jQoSnZJ3rr1q1L1VZf//rXU7XVO++8k6qtoNVyxg4A\nIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIO\nACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASbdI9AOxG\nr169UrXVCy+8kKqtunfvnpJ9fvzjH6dknxDC888/n6qtamtrU7JP//79U7JPCOHwww9P1VZx\nmzNnTqq2GjBgQEr2mTFjRkr2CSG8/fbbqdoKDgTO2AEARELYAQBEQtgBAERC2AEARELYAQBE\nQtgBAERC2AEARELYAQBEQtgBAERC2AEARELYAQBEQtgBAERC2AEARELYAQBEQtgBAERC2AEA\nRELYAQBEQtgBAERC2AEARELYAQBEQtgBAESiTboHgN1o3759qrbasGFDqrbq3r17qrZKlTPP\nPDPdI5B+t956a6q2+uCDD1Kyz9q1a1OyD/B5OWMHABAJYQcAEAlhBwAQCWEHABAJYQcAEAlh\nBwAQCWEHABAJYQcAEAlhBwAQiYz8zRMrV6586qmnamtrzzjjjHPPPbfRvT//+c9DCJMmTUrH\naAAAaZN5YXfttddOmzat/sORI0c++uijBx98cP0tN910UxB2AMCBJ8N+FHvvvfdOmzYtkUhc\ndNFFkydPHjRo0BNPPPHjH/843XMBAKRfhoXdrFmzQgi33nrrI488UlJSsnTp0rPPPvuuu+5a\nuXJlukcDAEizDAu78vLyEMJll12W/PDggw++//7727Zte+ONN6Z1LgCA9MuwsNuxY0cIoUuX\nLvW39O3b97rrrnvqqacWL16cvrkAANIvw8IuPz8/hLBixYqGN06YMKFLly7XX3/9zp070zQX\nAED6ZVjYjRw5MoTw/e9//7XXXqu/sXPnzlOmTFm+fPnll1+ePKUHAHAAyrCwmzRp0rHHHvvS\nSy/169cvefYuqbi4eOTIkb/73e+OOeaYNI4HAJBGGRZ2nTt3Xr58eUlJyRe+8IWampr62xOJ\nxKOPPjpp0qQtW7akcTwAgDTKsLALIXTq1Gny5Ml/+9vfPvnkk4a3H3zwwbfeeus//vGPV155\nZe7cuekaDwAgXTLvN0/sXVZW1oABAwYMGJDuQQAAWlpsYUcc3n777VRtdcstt6Rqq9tuuy0l\n+xx++OEp2ad1SuHLIWbPnp2qrb7yla+kaqtUmTp1aqq2+vvf/56qrVx/BpkuwrDr379/CGHV\nqlVNWbxz58558+bV1tbuZU1lZWVKBgMA2K8iDLuKioqmLy4rKxsxYsT+GwYAoMVEGHZlZWVN\nX1xUVDR37ty9n7GbPn36ggUL9nUsAID9LMKwGzJkSNMXZ2VlDR8+fO9r5s2bt08DAQC0iEwN\nu7Vr1y5atGj16tWbNm3Kzc3t3r17QUHB4MGDs7Ky0j0aAEB6ZF7YVVZWFhcX7/YsWl5e3tix\nYydPntyxY8eWHwwAIL0yLOzWrFlTWFhYXV3doUOHoUOHDhgwoGvXrolEYv369eXl5fPnz7/z\nzjtLS0sXL17cpUuXdA8LANCiMizsSkpKqqurR44cOXv27A4dOjS6d926dcOGDVu6dOmUKVPu\nuOOOtEwIAJAuGfYrxUpLS0MIU6dO3bXqQgh5eXkzZswIIcyZM6elJwMASLcMC7uampoQQu/e\nvfe0oF+/fiGEqqqqlpsJAKB1yLCwy8/PDyEsWbJkTwtWrFgRQujZs2fLzQQA0DpkWNiNGTMm\nhDB+/PjdvmPwsmXLxo0bF0IYPXp0Cw8GAJB2GXbxxMSJExcuXFhWVlZUVJSfnz9w4MBu3bqF\nEGpqalauXJn8pa6FhYUlJSVpHhQAoMVlWNi1b9++tLR05syZ06dPLy8vX7NmTcN7+/Tpc8UV\nV0yYMKFt27bpmhAAIF0yLOxCCNnZ2cXFxcXFxVVVVRUVFTU1Ndu2bevUqVNBQUFBQUG6pwMA\nSJvMC7t6PXr06NGjR7qnAABoLTLs4gkAAPZE2AEARELYAQBEQtgBAERC2AEARELYAQBEIoPf\n7oSIbd++PVVbVVdXp2qrUaNGpWSfFP5mlHfffTdVWz399NMp2ef5559PyT4hhA0bNqRqqw4d\nOqRqq1T5+OOP0z0CECFn7AAAIiHsAAAiIewAACIh7AAAIiHsAAAiIewAACIh7AAAIiHsAAAi\nIewAACIh7AAAIiHsAAAiIewAACIh7AAAIiHsAAAiIewAACIh7AAAIiHsAAAiIewAACIh7AAA\nIiHsAAAiIewAACLRJt0DwP717LPPpnuExi644IJ0j3DA+fjjj9M9AkBLcMYOACASwg4AIBLC\nDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACAS\nwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAg\nEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4A\nIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIO\nACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLC\nDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBJt0j1AM61du3bRokWrV6/etGlTbm5u9+7d\nCwoKBg8enJWVle7RAADSI/PCrrKysri4eN68ebvelZeXN3bs2MmTJ3fs2LHlBwMASK8MC7s1\na9YUFhZWV1d36NBh6NChAwYM6Nq1ayKRWL9+fXl5+fz58++8887S0tLFixd36dIl3cMCALSo\nDAu7kpKS6urqkSNHzp49u0OHDo3uXbdu3bBhw5YuXTplypQ77rgjLRMCAKRLhl08UVpaGkKY\nOnXqrlUXQsjLy5sxY0YIYc6cOS09GQBAumVY2NXU1IQQevfuvacF/fr1CyFUVVW13EwAAK1D\n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\nbsID26H/g9oAAAAASUVORK5CYII=",
   "text/plain": "Plot with title \u201cLabel for this image: 3\u201d"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

The proportion of each label within the training data is known to significantly affect the results of multi-class classification models.
We can see that in our MNIST training set, the number of images from each digit is fairly evenly distributed:

```{.python .input  n=4}
table(train_y)
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "train_y\n   0    1    2    3    4    5    6    7    8    9 \n4132 4684 4177 4351 4072 3795 4137 4401 4063 4188 "
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## Configuring a neural network

Now that we have the data, let’s create a neural network model. Here, we can use the ``Symbol`` framework in MXNet to declare our desired network architecture:

```{.python .input  n=5}
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
```

1)  ``data`` above represents the input data, i.e. the inputs to our neural network model.

2)  We define the first hidden layer with ``fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)``. 
This is a standard fully-connected layer that takes in ``data`` as its input, can be referenced by the name "fc1", and consists of 128 hidden neurons.

3)  The rectified linear unit ("relu") activation function is chosen for this first layer "fc1". 

4)  The second layer "fc2" is another fully-connected layer that takes the (post-relu) activations of the first hidden layer as its input, consists of 64 hidden neurons, and also employs relu activations. Note that when specifying which activation function to use for which layer, you must refer to the approporiate name of the layer. 

5)  Fully-connected layer "fc3" produces the outputs of our model (it is the ouput layer). Note that this layer employs 10 neurons (corresponding to 10 output values), one for each of the 10 possible classes in our classification task.  To ensure the output values represent valid class-probabilities (i.e. they are nonnegative and sum to 1), the network finally applies the softmax function to the outputs of "fc3".

## Training

We are almost ready to train the neural network we have defined. 
Before we start the computation, let’s decide which device to use:

```{.python .input  n=6}
devices <- mx.cpu()
```

This command tells **mxnet** to use the CPU for all neural network computations. 

Now, you can run the following command to train the neural network!

```{.python .input  n=7}
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train_x, y=train_y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                        epoch.end.callback=mx.callback.log.train.metric(100))
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Warning message in mx.model.select.layout.train(X, y):\n\u201cAuto detect layout input matrix, use colmajor..\n\u201dStart training with 1 devices\n[1] Train-accuracy=0.860404763388492\n[2] Train-accuracy=0.959642860435304\n[3] Train-accuracy=0.972404770198322\n[4] Train-accuracy=0.980047629986491\n[5] Train-accuracy=0.983952391147614\n[6] Train-accuracy=0.985880962581862\n[7] Train-accuracy=0.987619057155791\n[8] Train-accuracy=0.989380960521244\n[9] Train-accuracy=0.990476198707308\n[10] Train-accuracy=0.991595245259149\n"
 }
]
```

Note that ``mx.set.seed`` is the function that controls all randomness in **mxnet** and is critical to ensure reproducible results 
(R's ``set.seed`` function does not govern randomness within **mxnet**).

By declaring we are interested in ``mx.metric.accuracy`` in the above command, the loss function used during training is automatically chosen as the cross-entropy loss -- the de facto choice for multiclass classification tasks.

## Making Predictions

We can easily use our trained network to make predictions on the test data:

```{.python .input  n=8}
preds <- predict(model, test_x)
predicted_labels <- max.col(t(preds)) - 1
table(predicted_labels)
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Warning message in mx.model.select.layout.predict(X, model):\n\u201cAuto detect layout input matrix, use colmajor..\n\u201d"
 },
 {
  "data": {
   "text/plain": "predicted_labels\n   0    1    2    3    4    5    6    7    8    9 \n2825 3250 2764 2686 2715 2545 2750 2873 2787 2805 "
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

There are 28,000 test examples, and our model produces 10 numbers for each example, which represent the estimated probabilities of each class for a given image. ``preds`` is a 28000 x 10 matrix, where each column contains the predicted class-probabilities for a particular test image.  To determine which label our model estimates to be most likely for a test image (``predicted_labels``), we used ``max.col``.

 
Let's view a particular prediction:

```{.python .input  n=9}
i = 2  # change this to view the predictions for different test examples

class_probs_i = preds[,i]
names(class_probs_i) = as.character(0:9)
print("Predicted class probabilities for this test image:"); print(class_probs_i)
image(t(apply(matrix(test_x[,i],nrow=28, byrow=TRUE), 2, rev)) , col=gray((0:255)/255), 
      xlab="", ylab="", main=paste0("Predicted Label: ",predicted_labels[i], ".  Confidence: ", floor(max(class_probs_i)*100),"%"))
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[1] \"Predicted class probabilities for this test image:\"\n           0            1            2            3            4            5 \n9.999999e-01 8.445894e-11 3.864528e-10 7.600093e-14 5.628660e-14 1.228183e-07 \n           6            7            8            9 \n1.107560e-08 1.408887e-08 4.424089e-15 1.199879e-11 \n"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0gAAANICAIAAAByhViMAAAABmJLR0QA/wD/AP+gvaeTAAAg\nAElEQVR4nOzde3yU5Z3w/ysEBJIUxACGQ0qrUTwgRbdZbXHVSLW2Lw5WqrWsIloftU1Xrba4\nXRWLrdbdbsF6QGSrPUj18dBScZeStTW0gCDVFcUiQS1RtESFQYhyiJI8f9y/zSu/JOAQhkzm\n8v3+w5fMXLnmm9wT8uHOHPKampoCAAC5r1u2BwAAIDOEHQBAJIQdAEAkhB0AQCSEHQBAJIQd\nAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSE\nHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAk\nhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBA\nJIQdAEAkhB2xKSkpyWtPQUHByJEjb7zxxnfffbcTxnj33XeT2/3ggw+SS+644468vLwTTzyx\nE259z9rO1q6XX345WbZo0aJM3fRll12Wl5f3ve99L1MbhhDmzp2bl5f36U9/OoN77j/PPffc\nVVddNXLkyOLi4p49e5aUlJxxxhl33333zp07O2eArVu3XnzxxYMGDcrPz589e3ZI+56Z5t0G\nyC5hx0fF9u3bV61adcMNN/z93//922+/ne1x6CrefPPNq6+++sgjjywoKBg0aNDYsWN/97vf\n7Y8b+uCDD775zW8ed9xxM2fOXLVqVSqVamhoePPNN6uqqi677LKRI0e++OKL++N2W7n22mvv\nueeeurq6xsbGTri53PLEE09MmDDh4IMP7tmz57Bhw84+++ylS5fuy8oVK1acfvrphYWFffv2\nHT9+/AsvvNB2zbx58/r167dp06YMfzJ8ZDVBXA4++OAQwg9/+MOWF+7cufOvf/3rD3/4wx49\neoQQvvKVr+zvMerr65Nvsffff39f9snPzw8hPPvss5karCnt2V566aVkWXV1daZu+tJLLw0h\n3HDDDZnasKmp6b777gsh/N3f/V0HPvall14aNGhQ278Yr7322gxO2NTUtGvXrvHjxyebjxs3\n7ne/+92mTZsaGhpqa2vvvPPOAQMGhBAGDRq0fv36zN5uW5/61KdCCFdcccX27dv36gMzdZfu\nsv7t3/6t7T2hW7dut912W8dWPv744wcccEDLNYWFhStWrGi5pqGh4bDDDvv3f//3/f7p8ZEh\n7IhNu2HX7Nprrw0h9OjRo76+fr+OIeza1aXCbteuXcccc0wI4ZBDDnn88ce3bdv2xhtv/Mu/\n/EvyWc+bNy+DQ95yyy3JtnfeeWfba1966aW+ffuGEMaOHZvBG23XsGHDQgi//e1v9/YD4w67\nP/3pT3l5eSGEL3/5yytXrnz33XdffPHFb3zjG0mxPfXUU3u7srGx8ROf+ES3bt1mzJixadOm\nN998M7lrtbqjzpgx49BDD925c2enfrZETdgRmz2H3eLFi5MfTs8///x+HUPYtatLhd3//b//\nN4TQs2fPtWvXtrz8ggsuCCGMGjUqUxNu3ry5qKgohPB//s//2d2an/70p8lXe82aNZm63XYl\nYfff//3fe/uBcYfd6aefHkL4/Oc/3+ryr33ta60uT3NlTU1NCOHCCy9sueaMM84IIWzatCn5\n46ZNm/r16/fII49k+JPho81j7PiISv7NnTzu/swzz9y+fft11103bNiwT37yk81rNm/efN11\n1x1zzDGFhYUHHnjgZz/72dtuu63dB7kvXLjw9NNPP+igg4qKio477rh777237Zp2H6K+Zs2a\nCy64YOjQoT179vz4xz9++umnP/LII8mDn84444y8vLxdu3aFEI499ti8vLwlS5bsp9kyYtGi\nReeee+6hhx7au3fvIUOGnHTSSffee+/unhNQV1d32WWXDRkypFevXmVlZd/61rfeeuuttsvS\n/zRbWrhwYfIw/+XLl+9h2cMPPxxC+NKXvnTYYYe1vPwf//EfQwgrV67861//uucbStOvfvWr\nd999t1u3btddd93u1nzlK18pKCgIIVRXV7e8/JVXXrn00ksPOeSQXr169evX79RTT/3lL3/Z\n6uFxyV1rypQpjY2Nd95553HHHVdQUHDQQQd9/vOfb3mf+fKXv5yXl/fqq6+GEE4//fS8vLw9\nPHki/btNOscozQmb7eH7Yq9uN6R3Z2hoaEieIXTFFVe0uurCCy8MIVRXVydPukp/ZVNTU/jf\nv2d2Z/r06SNGjJg4ceIe1sBey3ZZQobt+Yxd8tuQ3r1779ixo+l/T/Z84Qtf+Id/+IfkO2LY\nsGHJyueff37w4MFtv2UOP/zw1157reWe7T7g5vLLL0/+p/n0xu233x5CGD16dPMHPvzww60e\ngpM499xzm5qaPv/5z7e6fPHixftptnbt1Rm7G264oe0NhRCOO+64lr/1Ts7YffWrXx04cGCr\nlYMHD37xxRdb7pnmp9n2jF3zsx+WLVu2u4EbGxuTR7bdc889ra7avn17z549Qwi/+tWvPvQT\nT0fyk/u4447b2w+cN29er1692n4FPve5z7377rvNy5K71le/+tUvfvGLrVZ27959+fLlLcdo\n6a677mpq756Z/t0mzWOU5oSJPX9f7NXtNqV3Z3jjjTeSNTU1Na2uev3115Orfv/73+/VysbG\nxmHDhuXn5//kJz9JpVJvvfXWtGnTQou7QU1NzQEHHPD000/vbiroGGFHbNoNu23btv3lL3/5\n9re/nfxm8+tf/3pyedIE3bp1692791133bVx48bk8lQqVVpaGkI49dRTFy9eXF9fn0qlHnnk\nkeTXWCeccEJjY2Oy8sknn0z+UX7WWWetWrVq586da9euTU75tPop2OrH5/PPP5/Uw5lnnrlq\n1aodO3a89tpr119/ffJRCxcuTJa1/VXs/pitXemH3YYNG5I5/+mf/umll17asWPH22+//eCD\nD/bv37/VsUjCLoRQUlJy3333bd68+d1333344YeTxjrmmGOah0//0+zYr2LffPPNZJInn3yy\n7bXJidsf/OAHe7Xn7owYMSKEcNFFF+3VR61atSqpurPOOuu5557bsWNHXV3dXXfd1adPn/D/\nfwJQctfKz8/v3r37zTffXFdXt3Pnzurq6uR74ayzzmq5bfIFbHlMW90z07/bpH+M0p8wne+L\n9G83TVu2bEn2b/63U7Nnn302uSqp/PRXNjU1LVy4MHm2VrOCgoLmih03btz555+/V3NCOoQd\nsUl+VOzBiBEj3nnnnWRx0gThf09dNEt+kIwePbpV+rz44otJwTQ/RDp5wM0Xv/jFVj9LvvSl\nL+057JIF5eXlu3btavmBkydPTgop+WPbsNsfs7Ur/bB79NFHQ3tnpJLTeC1PtCRh16tXr1Yn\nPKqqqpLbqqqq2ttPs2Nh1/zaIq+88krba//u7/4uhPDtb397r/bcneTc0ne+8529+qjkMH3m\nM59pdQ+ZP39+MvmqVauSS5K7Vgjhpz/9acuVd9xxRwjh4IMPbnnhh4Zd+neb9I9R+hOm832R\n/u2m74gjjggt/snX7Kqrrkomb37Ga/orm5qannzyyc997nMFBQVFRUVjx4597rnnksufeOKJ\ngoKCTngSNB9Bwo7Y7C7sevfufdRRR1177bVbtmxpXpw0QWFhYaufImVlZSGEuXPntt0/eSjS\njBkzmpqa3nnnnW7duoUQlixZ0mpZ84tatRt227dvT37Z9Mtf/nLPn07bsNsfs7VrH5880dDQ\n8OUvfzmEMHHixOYLk7D7x3/8x7brk5fh+O53v5v8Mc1Ps6mjYdf8iKs33nij7bWnnHJKCOGK\nK67Yqz13J2mpvXoJlW3btiUne9p9cu6hhx4aQvjRj36U/DG5ax188MGt7sbLli0LIXTv3r1l\nou057PbqbpP+MUpzwjS/L9K/3fTNmjUrhJCfn3/NNdfU1NQk5/iTu2ty/rI5SdNfuTu7du0a\nNWrUtGnT9nZISIcnTxCnto+xS/7+/cEPfpD8JqulsrKy5IdZor6+/uWXXw4hnHfeeW3fwSJ5\nrHfySP/nnnuusbGxW7du5eXlrfY87rjj9jDeX/7yl4aGhhDCySefvFefVyfM1mEvv/zy3Xff\n/Y1vfGPs2LHHHHNM8nS/dle2O8Cxxx4bQli/fn3Ym0+zw5p/R7Zt27a21yaPwU+ezbDviouL\nQwipVCr9D3nhhRfef//9sJt7SPIqLa2e29HqbhxCSJ6K+8EHHyRPwUlH+nebDhyjD50wne+L\n/XTfuOyyyyZNmrRr165//dd/HT58eEFBwdFHH3333XePGTMmOUWXPFpgr1buzs9+9rM333xz\n6tSpyR9//vOff+pTn+rZs+egQYO+8Y1v7NX9BNoSdhC6d+/e8o/p/MWa/ODfvHlzCKFXr15t\nH+vdq1ev5GRbu5pv4kN/cby7D9x/s3XAzp07L7300uHDh1922WV33XXXf/3Xf73wwgtHHXXU\nmWee2e76doMpqZ9k+PQ/zQ476KCDkv9555132l6bvA1Au69d3AHJCbZ233WgpU9/+tN5eXnf\n/OY3w/9+BQoLC/v169d2ZfIFbPUVaPdpFnsr/btNB47Rh06YzvfFfrpv5OXlzZ079+c///mJ\nJ57Yp0+fXr16HX300TfffPOjjz6aBHTznSH9le167733rr/++ptuuqmwsDCEcO2111544YXP\nP/98Q0ND8hjKz372s+3eJyFNwg5aa86OPfwKcsaMGc0rt23btnXr1labbN26dQ+nSZpPF7X9\nwKzP1gFTp06dM2dOY2PjxIkTZ8+evWjRorfeemvFihXJI9Xaavdnc/J8wyTv0v80O+zjH/94\n0hmrV69uddUHH3yQvCbIkUceuS830ayioiKEsHz58j2cSdqwYcP//M//hBCOP/748L8NtG3b\nth07drRdnDwBM3luSmalf7fZH8cone+L/XffyMvLu+CCCxYvXrxly5bt27e/8MIL3/3ud59/\n/vmdO3d+7GMfS84o7+3Ktm655ZaDDz44ea3ENWvW3HLLLSUlJY899ti7775bU1Mzbty4mpqa\nH/7whx2YHxLCDlrr379/cjrnueee2/PK5DcvIYQVK1a0umrPr6CWPEgohLBq1apWV33961/P\ny8tr+ypZnTZbB9x///0hhOuuu+6RRx659NJLTz755OS3UXV1de2ubztSU1PT008/HUIYNWpU\n2JtPs8O6det2wgknhDavGxdCePLJJ3fu3NmzZ8/PfOYzGbmtM888s1evXu+//367LyOS+PGP\nf9zU1FRYWDhhwoQQQvLSek1NTUnttVRfX598rfYcEB2T/t1mfxyjdL4v9tN9o7S0NC8vb968\nea0uf+ihh0IIY8aMaT6vn/7Ktl5//fUZM2bMmDEj+ZX0E0880djY+OMf/3js2LGFhYWHH374\nI4880rdv38cffzyDnxofNcIOWsvLyxs7dmwI4fbbb08e59Ssrq7uoIMOysvLe/LJJ0MIpaWl\nI0eODCH86Ec/armsaTevBNZs6NChyQc2P1uwef8HH3wwhHDqqadma7YOSM7AtXpU1oYNG5If\ndW0tXLiw+ZkZifnz57/88svdu3dPsib9T3NfnHvuuSGEBx98MHlgX6Lpf9/+60tf+lLyy7J9\nN2jQoMsuuyyEcOutt/76179uu+Dxxx//yU9+EkL4zne+kzwGdPDgwUnjNr8XWbN7771327Zt\nhYWFbV8Tbt+lf7fZH8cone+L/XTfOProo0MIyfO7m73yyivJO4JcdNFFHVjZ1j//8z+fdtpp\nyRnckN6LGMNe292pbMhRe36B4lZ294TKNWvWJL/x+cIXvvDss8/u3LnzjTfeuP/++5OXN6uo\nqGhe+dvf/jb5VpoyZcqaNWsaGhrWrl179tln9+jRI/m90u5e7qT5B/xVV1312muvbdu2benS\npcnj08vKyhoaGpJlSVv8x3/8R/MTG/fHbO1K/1mxSYKMGjXq6aefTl54bPbs2UOHDk0ep3XS\nSSc1NDQk8zc/eXDYsGGPPvroli1b6uvrH3jggeQczDe/+c0OHIIOv6XY9u3bk1NERx111J/+\n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\nzz77bG1t7YABA44//vgePXpkcTwAgGzJvbBr17HHHnvsscdmewoAgGyKJOw6bNeuXQsWLNix\nY8ce1tTW1nbWOAAAHRdh2B1xxBEhhDVr1qSzuLq6evz48ft5IgCAzhBh2NXU1KS/uKKiYv78\n+Xs+Yzdr1qxFixbt61gAAPtZhGFXXV2d/uL8/Pxx48btec2CBQv2bSIAgM4QYdidcsop2R4B\nACALcjXsNmzYsGTJknXr1tXX1xcWFg4cOLCsrGz06NH5+fnZHg0+0kpLSzO11QknnJCprSBk\n7m3Eq6qqMrIP7A+5F3a1tbWVlZXt/nq0uLj4/PPPnz59ep8+fTp/MACA7MqxsFu/fn15efnG\njRuLiorGjBkzYsSI/v375+Xlbd68efXq1QsXLrz11lurqqqWLl3ar1+/bA8LANCpcizspk2b\ntnHjxgkTJsydO7flm8YmNm3aNHbs2OXLl994440zZ87MyoQAANnSLdsD7J3kkQ0zZsxoW3Uh\nhOLi4tmzZ4cQ5s2b19mTAQBkW46FXSqVCiEMGTJkdwuGDx8eQqirq+u8mQAAuoYcC7vkCXfL\nli3b3YJnnnkmhDBo0KDOmwkAoGvIsbCbNGlSCGHKlCntvhXEU089NXny5BDCOeec08mDAQBk\nXY49eeKaa65ZvHhxdXV1RUVFaWnpqFGjBgwYEEJIpVIrV66sra0NIZSXl0+bNi3LgwIAdLoc\nC7uCgoKqqqo5c+bMmjVr9erV69evb3nt0KFDL7nkkqlTp/bs2TNbEwIAZEuOhV0IoUePHpWV\nlZWVlXV1dTU1NalUqqGhoW/fvmVlZWVlZdmeDgAga3Iv7JqVlJSUlJRkewoAgK4ix548AQDA\n7gg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASAg7AIBICDsAgEgIOwCASOTwW4oBlZWV\nmdpqwIABGdnniCOOyMg+IYSzzjorU1t1Qd26Zezf1Y2NjZnaKm6bNm3KyD6zZ8/OyD4hhEWL\nFmVqqz/+8Y+Z2oqc5owdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSE\nHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAkhB0AQCSEHQBAJIQdAEAk\nhB0AQCSEHQBAJLpnewDIGWeffXZG9pk4cWJG9gkhnHbaaZnaqm/fvhnZp1Eu++kAACAASURB\n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\nBwAQCWEHABAJYQcAEAlhBwAQCWEHABAJYQcAEInu2R4A2vGpT30qU1s999xzmdoqU0pLS7M9\nQju6dcvMP/MaGxszsk/08vPzsz1CbnjllVcytdWaNWsyss+ECRMysg/sD87YAQBEQtgBAERC\n2AEARELYAQBEQtgBAERC2AEARELYAQBEQtgBAERC2AEARELYAQBEQtgBAERC2AEARELYAQBE\nQtgBAERC2AEARELYAQBEQtgBAERC2AEARELYAQBEQtgBAERC2AEARKJ7tgeAdmzZsiVTWz30\n0EOZ2uqoo47qUvt0TY2NjdkeITds3bo1U1stXLgwU1t1Qd/5zncytdUbb7yRqa2gy3LGDgAg\nEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBI5+c4TK1eu/N3vfrdj\nx44TTzzxtNNOa3XtD37wgxDCddddl43RAACyJvfC7oorrrjtttua/zhhwoSHHnrogAMOaL7k\n+uuvD8IOAPjoybFfxd5999233XZbXl7eueeeO3369GOPPfbRRx/N4DsJAgDkrhwLu3vuuSeE\n8P3vf/+BBx6YNm3a8uXLTz311DvuuGPlypXZHg0AIMtyLOxWr14dQvja176W/PGAAw649957\ne/bs+S//8i9ZnQsAIPtyLOw++OCDEEK/fv2aLxk2bNiVV175u9/9bunSpdmbCwAg+3Is7EpL\nS0MIzzzzTMsLp06d2q9fv6uuumrXrl1ZmgsAIPtyLOwmTJgQQvjGN76xdu3a5gsPPPDAG2+8\nccWKFRdffHFySg8A4CMox8LuuuuuO/zww5977rnhw4cnZ+8SlZWVEyZM+PnPf37YYYdlcTwA\ngCzKsbA78MADV6xYMW3atKOPPjqVSjVfnpeX99BDD1133XXbt2/P4ngAAFmUY2EXQujbt+/0\n6dNfeOGF9957r+XlBxxwwPe///033nhj1apV8+fPz9Z4AADZknvvPLFn+fn5I0aMGDFiRLYH\nAQDobLGFHXF4/fXXM7XVpEmTMrXVySefnJF9TjrppIzsk1kDBgzIyD6XXXZZRvbJrIcffjgj\n+6xZsyYj+4QQNm7cmKmt7rrrrkxtBeS6CMPuiCOOCGn//btr164FCxbs2LFjD2tqa2szMhgA\nwH4VYdjV1NSkv7i6unr8+PH7bxgAgE4TYdhVV1env7iiomL+/Pl7PmM3a9asRYsW7etYAAD7\nWYRhd8opp6S/OD8/f9y4cXtes2DBgn0aCACgU+Rq2G3YsGHJkiXr1q2rr68vLCwcOHBgWVnZ\n6NGj8/Pzsz0aAEB25F7Y1dbWVlZWtnsWrbi4+Pzzz58+fXqfPn06fzAAgOzKsbBbv359eXn5\nxo0bi4qKxowZM2LEiP79++fl5W3evHn16tULFy689dZbq6qqli5d2q9fv2wPCwDQqXIs7KZN\nm7Zx48YJEybMnTu3qKio1bWbNm0aO3bs8uXLb7zxxpkzZ2ZlQgCAbMmxtxSrqqoKIcyYMaNt\n1YUQiouLZ8+eHUKYN29eZ08GAJBtORZ2qVQqhDBkyJDdLRg+fHgIoa6urvNmAgDoGnIs7EpL\nS0MIy5Yt292CZ555JoQwaNCgzpsJAKBryLGwS973c8qUKe2+YvBTTz01efLkEMI555zTyYMB\nAGRdjj154pprrlm8eHF1dXVFRUVpaemoUaOSdy5PpVIrV65M3tS1vLx82rRpWR4UAKDT5VjY\nFRQUVFVVzZkzZ9asWatXr16/fn3La4cOHXrJJZdMnTq1Z8+e2ZoQACBbcizsQgg9evSorKys\nrKysq6urqalJpVINDQ19+/YtKysrKyvL9nQAAFmTe2HXrKSkpKSkJNtTAAB0FTn25AkAAHZH\n2AEARELYAQBEQtgBAERC2AEARELYAQBEIodf7gQ62R//+McutU9m9enTJyP7dM3Pbvny5RnZ\n54033sjIPgD7iTN2AACREHYAAJEQdgAAkRB2AACREHYAAJEQdgAAkRB2AACREHYAAJEQdgAA\nkRB2AACREHYAAJEQdgAAkRB2AACREHYAAJEQdgAAkRB2AACREHYAAJEQdgAAkRB2AACREHYA\nAJEQdgAAkeie7QGALmHr1q0Z2efXv/51RvYBoAOcsQMAiISwAwCIhLADAIiEsAMAiISwAwCI\nhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMA\niISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLAD\nAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISw\nAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiE\n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\nl770pRDCpEmT/vrXv+7cufPVV1+98sorQwiDBw/uzMljlUqlnnjiiQsuuCBJtA8Nu5z7aS7s\n0nLUUUeFEJ544onmS+rr6wsLC/Py8v72t79l6kNIUwe+tkcffXQIoaampuWFX/nKV0IIN998\n8/4dN3Ydvqu/9957xx133LBhw377298KuwzqwBG57LLLQgj//M//3PLCr371qyGEFStW7N9x\nY9eBw3HccceFEF577bXmSxobGwsKCgoLC/f7uLHbvn17q3NvHxp2OffT3GPsPlxdXd3q1atL\nS0srKiqaLywqKvrc5z7X1NS0ZMmSjHwIaerY13bdunVDhgw5/PDDW16YfLvu2LFjvw4ctw7f\n1ZuamiZPnvzSSy899thjyRk7MqJjR+S///u/u3fvfvXVV7e88P77729qaiovL9+/E0etY4fj\ntNNOCyFMnTp1zZo1DQ0Nr7322pVXXrlt27axY8d20tzx6tWrV3MATZw48UPX5+JPc2H34Wpq\nakIIo0aNanX5iBEjQgjr1q3LyIeQpo59bd97771Wj3VtaGhIThSdeOKJ+2XQj4YO39Wvu+66\nefPm3X///cccc8x+nfCjpgNHJJVK/fWvfz3mmGN27dp13nnnFRcX9+7du7y8/Gc/+1lTU1Mn\nzByxjn2DTJ8+/fLLL3/ooYeOPPLInj17Dhs27Lbbbhs/fvxdd921vwemlVz8aS7sPlwqlQoh\nDBgwoNXl/fv3DyHU19dn5ENIU0a+tq+++uoXvvCFZ599dsKECck/jumYjh2OuXPn3nzzzT/6\n0Y+cgci4DhyRt99+O4RQXFz82c9+9le/+lUqldqxY8fTTz990UUXfe1rX9v/I8esY98gqVTq\nmWeeaWxsbHnhihUrnn/++f0zJruViz/Nhd2Ha2hoaPfyvLy8EML/a+/+Qapq4wCOP/bnWlwq\nIxAMqVAhbIiGiFoKWrLBIqyGqJCQpkpoaGiLhhpzcKmswbHdwqwblYFBaAbRENHQIET4h6gu\nFb3D5RWplzc9/il/fj7juT7w3POce+7Xw73n5vP5GRnCJE1z346MjJw/f76+vv7Bgwel/4ln\nfooLSYblePr0aUtLS0tLy9mzZ2d3cgtShhUZGRlJKfX09KxYseLeXwWWywAABLhJREFUvXsf\nP34cHh7u6OjI5XI3b958+PDhrE44tmznqwMHDvT29u7fv//58+efPn16/fp1a2vr0NDQoUOH\nSp3BnJmP7+bC7vdWrVqV/s32iYaHh1NKlZWVMzKEScq8b3/8+NHe3l5TU3Pp0qVdu3Y9e/as\nra0tl8vN6mzDy7Ac3d3dxWLx+vXrZf/asWNHSqmjo6OsrKyhoWH2Zx1ZhhUpLy9PKZWVlXV1\nde3evTufz1dUVJw4ceL06dMppZ6enlmfdFwZlqO/v7+vr6+2tvbWrVubN29evnx5bW3tlStX\nmpqa3r9/39XVNQfTZtx8fDcXdr9XV1eXUhoYGPhp+8uXL8cfnf4QJinbvv369WtTU9OpU6fq\n6uoeP358+/btXz8zQQYO9b9NhhWpqqpKKVVWVq5du3bi9q1bt6aURkdHZ2mqC0GG5Xj79m1K\nacuWLUuXLp24ffv27eOPMmfm5Slujr+FO09VV1enlPr6+sa3fPjwYeXKlWvWrPn27dtMDWGS\nMuzb0n3smpub7fwZN/1Dvb+/P7ndyczJsCLr1q1LKb169WrixtbW1uQexdM21eV49OhRSqmm\npqZYLE7cfvz48ZTSjRs3Zn3GC0bpW7G/vd3JvHs3d8VuUk6ePJlSOnLkyJMnT4rF4uDgYGNj\n49jY2LFjxxYvXjxTQ5ikqe7b79+/X716df369deuXbPzZ5xD/W+TYUVKX5LYt29fd3f32NjY\n0NBQW1tbe3t7Pp8/fPjwnM4+nKkux7Zt26qqqt68eXPw4MHBwcHPnz+/e/fuwoULnZ2d+Xy+\nsbFxzp/BQjf/TnF/uiznhy9fvvx6M6f6+vrR0dHSHxQKhZTSxo0bJz+EzKa6HC9evPifl8DF\nixf/3FOJIMOr4yeu2M2sbOernTt3/jRk0aJFpVvZMR0ZluPOnTvLli37dTk6Ozv/0JOI6T+v\n2AV4N3fFblLKy8sLhcK5c+c2bNhQ+knmM2fO9Pb2/vpzftMZwiRNdd/6VMqscqj/bbKdr+7e\nvXv58uVNmzblcrmKioqGhoZCoVD68QmmI8Ny7NmzZ2BgoLm5ubq6esmSJatXr967d+/9+/eP\nHj06lzOnZN6d4sp+uP8kAEAIrtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISw\nAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2\nAABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIO\nACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgB\nAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsA\ngCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcA\nEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAA\nghB2AABBCDsAgCD+AXv/Hjj4qFVPAAAAAElFTkSuQmCC",
   "text/plain": "Plot with title \u201cPredicted Label: 0.  Confidence: 99%\u201d"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

With a little extra effort, we can create a CSV-formatted file that contains our predictions for all of the test data.
After running the below command, you can submit the ``submission.csv`` file to the [Kaggle competition](https://www.kaggle.com/c/digit-recognizer/submit)!

```{.python .input  n=10}
submission <- data.frame(ImageId=1:ncol(test_x), Label=predicted_labels)
write.csv(submission, file='submission.csv', row.names=FALSE,  quote=FALSE)
```

## Convolutional Neural Network (LeNet)

Previously, we used a standard feedforward neural network (with only fully-connected layers) as our classification model.
For the same task, we can instead use a convolutional neural network, which employs alternative types of layers better-suited for handling the spatial structure present in image data. 
The specific convolutional network we use is the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture, which was previously proposed by Yann LeCun for recognizing handwritten digits.

Here's how we can construct the LeNet network:

```{.python .input  n=10}
# declare input data
data <- mx.symbol.Variable('data')
# first convolutional layer: 20 filters
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# second convolutional layer: 50 filters
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# flatten resulting outputs into a vector
flatten <- mx.symbol.Flatten(data=pool2)
# first fully-connected layer
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fully-connected layer (output layer)
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
lenet <- mx.symbol.SoftmaxOutput(data=fc2)
```

This model first passes the input through two convolutional layers with max-pooling and tanh activations, before subsequently applying two standard fully-connected layers (again with tanh activation). 
The number of filters employed by each convolutional layer can be thought of as the number of distinct patterns that the layer searches for in its input.
In convolutional neural networks, it is important to *flatten* the spatial output of convolutional layers into a vector before passing these values to subsequent fully-connected layers.

We also reshape our data matrices into spatially-arranged arrays, which is important since convolutional layers are highly sensitive to the spatial layout of their inputs.

```{.python .input  n=11}
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))
```

Before training our convolutional network, we once again specify what devices to run computations on.

```{.python .input  n=18}
n.gpu <- 1 # you can set this to the number of GPUs available on your machine

device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
```

We can pass a list of devices to ask MXNet to train on multiple GPUs (you can do this for CPUs, but because internal computation of CPUs is already multi-threaded, there is less gain than with using GPUs).

Start by training our convolutional neural network on the CPU first. Because this takes a bit time, we run it for just one training epoch:

```{.python .input  n=13}
mx.set.seed(0)
model <- mx.model.FeedForward.create(lenet, X=train_array, y=train_y,
            ctx=device.cpu, num.round=1, array.batch.size=100,
            learning.rate=0.05, momentum=0.9, wd=0.00001,
            eval.metric=mx.metric.accuracy,
            epoch.end.callback=mx.callback.log.train.metric(100))
```

Here, ``wd`` specifies a small amount of weight-decay (i.e. l2 regularization) should be employed while training our network parameters.

We could also train the same model using the GPU instead, which can significantly speed up training. 

**Note:** The below command that specifies GPU training will only work if the GPU-version of MXNet has been properly installed. To avoid issues, we set the Boolean flag ``use_gpu`` based on whether or not a GPU is detected in the current environment.

```{.python .input  n=20}
use_gpu <- !inherits(try(mx.nd.zeros(1,mx.gpu()), silent = TRUE), 'try-error') # TRUE if GPU is detected.
if (use_gpu) {
    mx.set.seed(0)
    model <- mx.model.FeedForward.create(lenet, X=train_array, y=train_y,
                ctx=device.gpu, num.round=5, array.batch.size=100,
                learning.rate=0.05, momentum=0.9, wd=0.00001,
                eval.metric=mx.metric.accuracy,
                epoch.end.callback=mx.callback.log.train.metric(100))
}
```

Finally, we can submit the convolutional neural network predictions to Kaggle to see if our ranking in the competition has improved:

```{.python .input  n=15}
preds <- predict(model, test_array)
predicted_labels <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test_x), Label=predicted_labels)
write.csv(submission, file='lenetsubmission.csv', row.names=FALSE, quote=FALSE)
```

## User Exercise 

Try to further improve MNIST classification performance by playing with factors such as:

- the neural network architecture (# of convolutional/fully-connected layers, # of neurons in each layer, the activation functions and pooling-strategy, etc.)

- the type of optimizer (c.f. ``mx.opt.adam``) used and its hyperparameters (e.g. learning rate, momentum)

- how the neural network parameters are initialized (c.f. ``mx.init.normal``)

- different regularization strategies (e.g. altering the value of ``wd`` or introducing dropout: ``mx.symbol.Dropout``)

- augmenting the training data with additional examples created through simple transformations such as rotation/cropping
