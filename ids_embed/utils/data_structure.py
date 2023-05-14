def flatten(list_of_lists):
    """
    flatten a list of lists recursively
    """
    if list_of_lists is None:
        return []
    if not isinstance(list_of_lists, list):
        return [list_of_lists]
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def list_reverse(l):
    return list(reversed(l))
