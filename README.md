# Analyze pollutants in arctic cod and haddock

Some colleagues have measured pollutants (PCB, DDT, HCB, etc) in
arctic cod and haddock.  I was asked to assist with data analysis and
plotting for the paper, and decided this was a good opportunity to
learn more about the Python libraries 'pandas' and 'matplotlib'.

In the end, my program reads the data as CSV files, and outputs plots
and a LaTeX source file that produces the supplementary document as a
PDF.  I'm particularly happy about the code that inspects the output
from 'statsmodels' and converts it into sensible units.  The rather
obscure regression parameters in a log-transformred domain is
converted into a percentage-wise decline per year, or a
percentage-wise increase in pollutant concentration from doubling fish
size.

And the happy news is that as expected (but quite undercommunicated, I
think), pollutant levels have decreased substantially in the last
decades. 
