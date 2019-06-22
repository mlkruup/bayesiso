class Pair:

    def __init__(self, sc, cl):
        self.sc = sc # score from the inital model
        self.csc = cl
        self.cl = cl # true class
        self.nr0 = None
        self.nr1 = None


    def set_class_count(self, nr0, nr1):
        self.nr0 = nr0
        self.nr1 = nr1

    def __repr__(self):
        return ",".join([str(x) for x in [self.sc, self.cl]])


def join_pairs_with_same_score(pairs, grouping_method="repl", return_pairs = False):
    current_pairs = [pairs[0][0]]
    new_pairs = {}
    c = 0
    for i in range(1, len(pairs)):
        if pairs[i][0].sc == current_pairs[-1].sc:
            # if np.isclose(pairs[i][0].sc, current_pairs[-1].sc):
            current_pairs.append(pairs[i][0])
        else:
            nr_1 = sum([p.cl for p in current_pairs])
            nr_all = len(current_pairs)
            nr_0 = nr_all - nr_1
            new_score = nr_1 / len(current_pairs)

            if grouping_method == "repl":
                new_pairs[c] = current_pairs
                for pair in new_pairs[c]:
                    pair.csc = new_score
            elif grouping_method == "join":
                new_pairs[c] = [Pair(current_pairs[0].sc, new_score)]
                new_pairs[c][0].set_class_count(nr_0, nr_1)
            current_pairs = [pairs[i][0]]
            c += 1
    if len(current_pairs) > 0:

        nr_1 = sum([p.cl for p in current_pairs])
        nr_all = len(current_pairs)
        nr_0 = nr_all - nr_1
        new_score = nr_1 / len(current_pairs)
        if grouping_method == "repl":
            new_pairs[c] = current_pairs
            for pair in new_pairs[c]:
                pair.csc = new_score
        elif grouping_method == "join":
            new_pairs[c] = [Pair(current_pairs[0].sc, new_score)]
            new_pairs[c][0].set_class_count(nr_0, nr_1)

    # diff part from isotonic regression

    if return_pairs:
        return new_pairs

    scores = []
    classes = []
    nr0 = []
    nr1 = []

    a = -float("inf")  # -math.inf
    for x in new_pairs:
        assert new_pairs[x][0].sc == new_pairs[x][-1].sc
        for y in new_pairs[x]:
            scores.append(y.sc)
            classes.append(y.csc)
            if y.nr0 is None:
                if y.cl == 1:
                    nr0.append(0)
                    nr1.append(1)
                elif y.cl == 0:
                    nr0.append(1)
                    nr1.append(0)
            else:
                nr0.append(y.nr0)
                nr1.append(y.nr1)
            try:
                assert y.sc >= a
            except:
                print(y.sc, a)
                raise AssertionError
            a = y.sc

    return scores, classes, nr0, nr1