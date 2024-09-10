def merge_intervals(intervals):
    # If there are no intervals, return an empty list
    if not intervals:
        return []
    
    # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])
    
    # Initialize the merged list with the first interval
    merged = [intervals[0]]
    
    for i in range(1, len(intervals)):
        # Compare the current interval's start with the last merged interval's end
        if intervals[i][0] <= merged[-1][1]:
            # If they overlap, merge them by updating the end of the last merged interval
            merged[-1][1] = max(merged[-1][1], intervals[i][1])
        else:
            # Otherwise, add the current interval to the merged list
            merged.append(intervals[i])
    
    return merged

# Test case
intervals = [[1,3],[2,6],[8,10],[15,18]]
result = merge_intervals(intervals)
print(result)
