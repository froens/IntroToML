#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    zipped = zip(predictions, ages, net_worths)
    sorted_by_re = sorted([(p[0], a[0], nw[0], abs(p[0]-nw[0])) for (p, a, nw) in zipped], key=lambda (p, a, nw, re): re)

    ninety_pct = len(sorted_by_re)-(len(sorted_by_re)/10)
    sorted_by_re = sorted_by_re[:ninety_pct]
    cleaned_data = [(age, nw, re) for (pre, age, nw, re) in sorted_by_re]

    ### your code goes here
    return cleaned_data
