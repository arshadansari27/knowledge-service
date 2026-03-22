% Transitive link (bounded to 2-hop to prevent runaway)
indirect_link(A, P, C) :-
    claims(A, P, B, _),
    claims(B, P, C, _),
    A \= C.

% Cross-predicate causal chains: A causes B, B increases/decreases C
causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, increases, C, _).

causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, decreases, C, _).
