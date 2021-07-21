# 0x02. Calculus
==============

*   _US Program > Trimester 3 - Machine Learning > MM1000_
*   By Alexa Orrico, Software Engineer at Holberton School=

## Resources
---------

**Read or watch**:

*   [Sigma Notation](https://www.youtube.com/watch?v=TjMLzklnn2c "Sigma Notation") (_starting at 0:32_)
*   [Π Product Notation](https://www.youtube.com/watch?v=sP1-EQJKSgk "Π Product Notation") (_up to 0:20_)
*   [Sigma and Pi Notation](https://mathmaine.com/2010/04/01/sigma-and-pi-notation/ "Sigma and Pi Notation")
*   [What is a Series?](https://virtualnerd.com/algebra-2/sequences-series/define/defining-series/series-definition "What is a Series?")
*   [What is a Mathematical Series?](https://www.quickanddirtytips.com/education/math/what-is-a-mathematical-series "What is a Mathematical Series?")
*   [List of mathematical series: Sums of powers](https://en.wikipedia.org/wiki/List_of_mathematical_series#Sums_of_powers "List of mathematical series: Sums of powers")
*   [Bernoulli Numbers(Bn)](https://en.wikipedia.org/wiki/Bernoulli_number "Bernoulli Numbers(Bn)")
*   [Bernoulli Polynomials(Bn(x))](https://en.wikipedia.org/wiki/Bernoulli_polynomials "Bernoulli Polynomials(Bn(x))")
*   [Derivative (mathematics)](https://simple.wikipedia.org/wiki/Derivative_%28mathematics%29 "Derivative (mathematics)")
*   [Calculus for ML](https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html "Calculus for ML")
*   [1 of 2: Seeing the big picture](https://www.youtube.com/watch?v=tt2DGYOi3hc "1 of 2: Seeing the big picture")
*   [2 of 2: First Principles](https://www.youtube.com/watch?v=50Bda5VKbqA "2 of 2: First Principles")
*   [1 of 2: Finding the Derivative](https://www.youtube.com/watch?v=fXYhyyJpFe8 "1 of 2: Finding the Derivative")
*   [2 of 2: What do we discover?](https://www.youtube.com/watch?v=Un0RcTMPJ64 "2 of 2: What do we discover?")
*   [Deriving a Rule for Differentiating Powers of x](https://www.youtube.com/watch?v=I8IM9P-2TRU "Deriving a Rule for Differentiating Powers of x")
*   [1 of 3: Introducing a substitution](https://www.youtube.com/watch?v=U0m4MsOgETw "1 of 3: Introducing a substitution")
*   [2 of 3: Combining derivatives](https://www.youtube.com/watch?v=z-tEsz0bSrA "2 of 3: Combining derivatives")
*   [How To Understand Derivatives: The Product, Power & Chain Rules](https://betterexplained.com/articles/derivatives-product-power-chain/ "How To Understand Derivatives: The Product, Power & Chain Rules")
*   [Product Rule](https://en.wikipedia.org/wiki/Product_rule "Product Rule")
*   [Common Derivatives and Integrals](https://www.coastal.edu/media/academics/universitycollege/mathcenter/handouts/calculus/deranint.PDF "Common Derivatives and Integrals")
*   [Introduction to partial derivatives](https://mathinsight.org/partial_derivative_introduction "Introduction to partial derivatives")
*   [Partial derivatives - How to solve?](https://www.youtube.com/watch?v=rnoToCoEK48 "Partial derivatives - How to solve?")
*   [Integral](https://en.wikipedia.org/wiki/Integral "Integral")
*   [Integration and the fundamental theorem of calculus](https://www.youtube.com/watch?v=rfG8ce4nNh0 "Integration and the fundamental theorem of calculus")
*   [Introduction to Integration](https://www.mathsisfun.com/calculus/integration-introduction.html "Introduction to Integration")
*   [Indefinite Integral - Basic Integration Rules, Problems, Formulas, Trig Functions, Calculus](https://www.youtube.com/watch?v=o75AqTInKDU "Indefinite Integral - Basic Integration Rules, Problems, Formulas, Trig Functions, Calculus")
*   [Definite Integrals](https://www.mathsisfun.com/calculus/integration-definite.html "Definite Integrals")
*   [Definite Integral](https://www.youtube.com/watch?v=Gc3QvUB0PkI "Definite Integral")
*   [Multiple integral](https://en.wikipedia.org/wiki/Multiple_integral "Multiple integral")
*   [Double integral 1](https://www.youtube.com/watch?v=85zGYB-34jQ "Double integral 1")
*   [Double integrals 2](https://www.youtube.com/watch?v=TdLD2Zh-nUQ "Double integrals 2")

## Learning Objectives
-------------------
### General

*   Summation and Product notation
*   What is a series?
*   Common series
*   What is a derivative?
*   What is the product rule?
*   What is the chain rule?
*   Common derivative rules
*   What is a partial derivative?
*   What is an indefinite integral?
*   What is a definite integral?
*   What is a double integral?

## Requirements
------------

### Multiple Choice Questions

*   Allowed editors: `vi`, `vim`, `emacs`
*   Type the number of the correct answer in your answer file
*   All your files should end with a new line

Example:

What is 9 squared?

1.  99
2.  81
3.  3
4.  18
```
    alexa@ubuntu$ cat answer_file
    2
    alexa@ubuntu$
```

### Python Scripts

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.5)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module
*   All your files must be executable
*   The length of your files will be tested using `wc`

## Tasks
-----

### 0\. Sigma is for Sum
![](https://latex.codecogs.com/gif.latex?\sum_{i=2}^{5}&space;i "\sum_{i=2}^{5} i")
1.  3 + 4 + 5
2.  3 + 4
3.  2 + 3 + 4 + 5
4.  2 + 3 + 4

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `0-sigma_is_for_sum`


### 1\. The Greeks pronounce it sEEgma
![](https://latex.codecogs.com/gif.latex?\sum_{k=1}^{4}&space;9i&space;-&space;2k "\sum_{k=1}^{4} 9i - 2k")
1.  90 - 20
2.  36i - 20
3.  90 - 8k
4.  36i - 8k

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `1-seegma`


### 2\. Pi is for Product
![](https://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;1}^{m}&space;i "\prod_{i = 1}^{m} i")
1.  (m - 1)!
2.  0
3.  (m + 1)!
4.  m!

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `2-pi_is_for_product`


### 3\. The Greeks pronounce it pEE
![](https://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;0}^{10}&space;i "\prod_{i = 0}^{10} i")
1.  10!
2.  9!
3.  100
4.  0

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `3-pee`


### 4\. Hello, derivatives!

![](https://latex.codecogs.com/gif.latex?\frac{dy}{dx} "\frac{dy}{dx}") where ![](https://latex.codecogs.com/gif.latex?y&space;=&space;x^4&space;+&space;3x^3&space;-&space;5x&space;+&space;1 "y = x^4 + 3x^3 - 5x + 1")

1.  ![](https://latex.codecogs.com/gif.latex?3x^3&space;+&space;6x^2&space;-4 "3x^3 + 6x^2 -4")
2.  ![](https://latex.codecogs.com/gif.latex?4x^3&space;+&space;6x^2&space;-&space;5 "4x^3 + 6x^2 - 5")
3.  ![](https://latex.codecogs.com/gif.latex?4x^3&space;+&space;9x^2&space;-&space;5 "4x^3 + 9x^2 - 5")
4.  ![](https://latex.codecogs.com/gif.latex?4x^3&space;+&space;9x^2&space;-&space;4 "4x^3 + 9x^2 - 4")

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `4-hello_derivatives`


### 5\. A log on the fire
![](https://latex.codecogs.com/gif.latex?\frac{d&space;(xln(x))}{dx} "\frac{d (xln(x))}{dx}")
1.  ![](https://latex.codecogs.com/gif.latex?ln(x) "ln(x)")
2.  ![](https://latex.codecogs.com/gif.latex?\frac{1}{x} + 1 "\frac{1}{x} + 1")
3.  ![](https://latex.codecogs.com/gif.latex?ln(x) + 1 "ln(x) + 1")
4.  ![](https://latex.codecogs.com/gif.latex?\frac{1}{x} "\frac{1}{x}")

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `5-log_on_fire`


### 6\. It is difficult to free fools from the chains they revere
![](https://latex.codecogs.com/gif.latex?\frac{d&space;(ln(x^2))}{dx} "\frac{d (ln(x^2))}{dx}")
1.  ![](https://latex.codecogs.com/gif.latex?\frac{2}{x} "\frac{2}{x}")
2.  ![](https://latex.codecogs.com/gif.latex?\frac{1}{x^2} "\frac{1}{x^2}")
3.  ![](https://latex.codecogs.com/gif.latex?\frac{2}{x^2} "\frac{2}{x^2}")
4.  ![](https://latex.codecogs.com/gif.latex?\frac{1}{x} "\frac{1}{x}")
5.  
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `6-voltaire`


### 7\. Partial truths are often more insidious than total falsehoods
![](https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;y}&space;f(x,&space;y) "\frac{\partial f(x, y)}{\partial y}") where ![](https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;e^{xy} "f(x, y) = e^{xy}") and ![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0 "\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0")
1.  ![](https://latex.codecogs.com/gif.latex?e^{xy} "e^{xy}")
2.  ![](https://latex.codecogs.com/gif.latex?ye^{xy} "ye^{xy}")
3.  ![](https://latex.codecogs.com/gif.latex?xe^{xy} "xe^{xy}")
4.  ![](https://latex.codecogs.com/gif.latex?e^{x} "e^{x}")

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `7-partial_truths`


### 8\. Put it all together and what do you get?
![](https://latex.codecogs.com/gif.latex?\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y}) "\frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y})") where ![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0 "\frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0")
1.  ![](https://latex.codecogs.com/gif.latex?2x(1+y)e^{x^2y} "2x(1+y)e^{x^2y}")
2.  ![](https://latex.codecogs.com/gif.latex?xe^{xy} "2xe^{2x}")
3.  ![](https://latex.codecogs.com/gif.latex?2x(1+x^2y)e^{x^2y} "2x(1+x^2y)e^{x^2y}")
4.  ![](https://latex.codecogs.com/gif.latex?e^{2x} "e^{2x}")

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `8-all-together`


### 9\. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
Write a function `def summation_i_squared(n):` that calculates ![](https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}&space;i^2 "\sum_{i=1}^{n} i^2"):
*   `n` is the stopping condition
*   Return the integer value of the sum
*   If `n` is not a valid number, return `None`
*   You are not allowed to use any loops
```
    alexa@ubuntu:0x02-calculus$ cat 9-main.py 
    #!/usr/bin/env python3
    
    summation_i_squared = __import__('9-sum_total').summation_i_squared
    
    n = 5
    print(summation_i_squared(n))
    alexa@ubuntu:0x02-calculus$ ./9-main.py 
    55
    alexa@ubuntu:0x02-calculus$
```

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `9-sum_total.py`


### 10\. Derive happiness in oneself from a good day's work
Write a function `def poly_derivative(poly):` that calculates the derivative of a polynomial:
*   `poly` is a list of coefficients representing a polynomial
    *   the index of the list represents the power of `x` that the coefficient belongs to
    *   Example: if ![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;x^3&space;+&space;3x&space;+5 "f(x) = x^3 + 3x +5"), `poly` is equal to `[5, 3, 0, 1]`
*   If `poly` is not valid, return `None`
*   If the derivative is `0`, return `[0]`
*   Return a new list of coefficients representing the derivative of the polynomial
```
    alexa@ubuntu:0x02-calculus$ cat 10-main.py 
    #!/usr/bin/env python3
    
    poly_derivative = __import__('10-matisse').poly_derivative
    
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))
    alexa@ubuntu:0x02-calculus$ ./10-main.py 
    [3, 0, 3]
    alexa@ubuntu:0x02-calculus$
```

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `10-matisse.py`


### 11\. Good grooming is integral and impeccable style is a must
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/ada047ad4cbee23dfed8.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7a9f3611da0f2a8d3b3f8d1a59ebc956014fc387ed6ca30ff90e57b2e9506116)
1.  3x2 + C
2.  x4/4 + C
3.  x4 + C
4.  x4/3 + C

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `11-integral`


### 12\. We are all an integral part of the web of life
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/9ed107b0dcdde8dd49ac.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e48c3ee411c5322290619edab940e3512c2d2e9736708e57c053d33718ab20a1)
1.  e2y + C
2.  ey + C
3.  e2y/2 + C
4.  ey/2 + C

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `12-integral`


### 13\. Create a definite plan for carrying out your desire and begin at once
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/b94ec3cf3ae61acd0275.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f63489b1b1d036c5e159e58ef10fb5aa5188af1defb9d230b7701b910cf718d8)
1.  3
2.  6
3.  9
4.  27

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `13-definite`


### 14\. My talents fall within definite limitations
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/44057bed4938503a9978.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=960670ef598fa6a530f95af99215b235a8895d6d5f19c189adce6ba8b21b31ee)
1.  \-1
2.  0
3.  1
4.  undefined

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `14-definite`


### 15\. Winners are people with definite purpose in life
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/3d88d653f3ba869b43b1.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2ac282e46f2b7848f3674f63a0029045881bceb87dea37cdaf38c1a2f4ce29e2)
1.  5
2.  5x
3.  25
4.  25x

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `15-definite`


### 16\. Double whammy
![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/a2409c32448118661d05.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210721%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210721T012315Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a8fd7f002559eebcac0e98000921095c2044f45c8d2dab8f1de0f2558c812d60)
1.  9ln(2)
2.  9
3.  27ln(2)
4.  27

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `16-double`


### 17\. Integrate
Write a function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial:
*   `poly` is a list of coefficients representing a polynomial
    *   the index of the list represents the power of `x` that the coefficient belongs to
    *   Example: if ![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;x^3&space;+&space;3x&space;+5 "f(x) = x^3 + 3x +5"), `poly` is equal to `[5, 3, 0, 1]`
*   `C` is an integer representing the integration constant
*   If a coefficient is a whole number, it should be represented as an integer
*   If `poly` or `C` are not valid, return `None`
*   Return a new list of coefficients representing the integral of the polynomial
*   The returned list should be as small as possible
```
    alexa@ubuntu:0x02-calculus$ cat 17-main.py 
    #!/usr/bin/env python3
    
    poly_integral = __import__('17-integrate').poly_integral
    
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
    alexa@ubuntu:0x02-calculus$ ./17-main.py 
    [0, 5, 1.5, 0, 0.25]
    alexa@ubuntu:0x02-calculus$
```

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x02-calculus`
*   File: `17-integrate.py`