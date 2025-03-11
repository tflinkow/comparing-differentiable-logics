from collections import namedtuple

Formula = namedtuple('Formula', 'description value')

groups: dict[str, list[Formula]] = {
    'Axiom schemata':
    [
        Formula(r'P \implies (Q \implies P)',
                lambda l, P, Q: l.IMPL(P, l.IMPL(Q, P))),

        Formula(r'(P \implies (Q \implies R)) \implies ( (P \implies Q) \implies (P \implies R) )',
                lambda l, P, Q, R: l.IMPL(l.IMPL(P, l.IMPL(Q, R)), l.IMPL(l.IMPL(P, Q), l.IMPL(P, R)))),

        Formula(r'(\lnot P \implies \lnot Q) \implies (Q \implies P)',
                lambda l, P, Q: l.IMPL(l.IMPL(l.NOT(P), l.NOT(Q)), l.IMPL(Q, P)))
    ],
    'Primitive propositions':
    [
        Formula(r'(P \vee P) \implies P',
                lambda l, P: l.IMPL(l.OR(P, P), P)),

        Formula(r'Q \implies (P \vee Q)',
                lambda l, P, Q: l.IMPL(Q, l.OR(P, Q))),

        Formula(r'(P \vee Q) \implies (Q \vee P)',
                lambda l, P, Q: l.IMPL(l.OR(P, Q), l.OR(Q, P))),

        Formula(r'(P \vee (Q \vee R)) \implies (Q \vee (P \vee R))',
                lambda l, P, Q, R: l.IMPL(l.OR(P, l.OR(Q, R)), l.OR(Q, l.OR(P, R)))),

        Formula(r'(Q \implies R) \implies ( (P \vee Q) \implies (P \vee R))',
                lambda l, P, Q, R: l.IMPL(l.IMPL(Q, R), l.IMPL(l.OR(P, Q), l.OR(P, R))))
    ],
    'Law of excluded middle':
    [
        Formula(r'P \vee \lnot P',
                lambda l, P: l.OR(P, l.NOT(P))),
    ],
    'Law of contradiction':
    [
        Formula(r'\lnot(P \wedge \lnot P)',
                lambda l, P: l.NOT(l.AND(P, l.NOT(P)))),
    ],
    'Law of double negation':
    [
        Formula(r'P \iff \lnot (\lnot P))',
                    lambda l, P: l.EQUIV(P, l.NOT(l.NOT(P)))),
    ],
    'Principles of transposition':
    [
        Formula(r'(P \iff Q) \iff (\lnot P \iff \lnot Q)',
                lambda l, P, Q: l.EQUIV(l.EQUIV(P, Q), l.EQUIV(l.NOT(P), l.NOT(Q)))),

        Formula(r'( (P \wedge Q) \implies R ) \iff ( (P \wedge \lnot R) \implies \lnot Q)',
                lambda l, P, Q, R: l.EQUIV(l.IMPL(l.OR(P, Q), R), l.IMPL(l.AND(P, l.NOT(R)), l.NOT(Q))))
    ],
    'Laws of tautology':
    [
        Formula(r'P \iff (P \wedge P)',
                lambda l, P: l.EQUIV(P, l.AND(P, P))),

        Formula(r'P \iff (P \vee P)',
                lambda l, P: l.EQUIV(P, l.OR(P, P)))
    ],
    'Laws of absorption':
    [
        Formula(r'(P \implies Q) \iff (P \iff (P \wedge Q))',
                lambda l, P, Q: l.EQUIV(l.IMPL(P, Q), l.EQUIV(P, l.AND(P, Q)))),

        Formula(r'Q \implies (P \iff (P \wedge Q))',
                lambda l, P, Q: l.IMPL(Q, l.EQUIV(P, l.AND(P, Q)))),
    ],
    'Assoc., comm., dist. laws':
    [
        Formula(r'(P \wedge (Q \vee R)) \iff ( (P \wedge Q) \vee (P \wedge R) )',
                lambda l, P, Q, R: l.EQUIV(l.AND(P, l.OR(Q, R)), l.OR(l.AND(P, Q), l.AND(P, R)))),

        Formula(r'(P \vee (Q \wedge R)) \iff ( (P \vee Q) \wedge (P \vee R) )',
                lambda l, P, Q, R: l.EQUIV(l.OR(P, l.AND(Q, R)), l.AND(l.OR(P, Q), l.OR(P, R)))),
    ],
    "De Morgan's laws":
    [
        Formula(r'\lnot(P \wedge Q) \iff (\lnot P \vee \lnot Q)',
                lambda l, P, Q: l.EQUIV(l.NOT(l.AND(P, Q)), l.OR(l.NOT(P), l.NOT(Q)))),

        Formula(r'\lnot(P \vee Q) \iff (\lnot P \wedge \lnot Q)',
                lambda l, P, Q: l.EQUIV(l.NOT(l.OR(P, Q)), l.AND(l.NOT(P), l.NOT(Q)))),
    ],
    'Material excluded middle':
    [
        Formula(r'(P \implies Q) \vee (Q \implies P)',
                lambda l, P, Q: l.OR(l.IMPL(P, Q), l.IMPL(Q, P))),
    ],
}