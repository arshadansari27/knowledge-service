% Core contradiction detection
% Claims use 4-arity (subject, predicate, object, source) to allow
% ProbLog to assign independent probabilities to the same ground triple.
contradicts(S, P1, O, P2) :-
    claims(S, P1, O, _),
    claims(S, P2, O, _),
    opposite(P1, P2).

% Same predicate, different object = potential value conflict
value_conflict(S, P, O1, O2) :-
    claims(S, P, O1, _),
    claims(S, P, O2, _),
    O1 \= O2.

% Multi-source support: a claim is "supported" if it appears from any source.
% ProbLog computes the combined probability via Noisy-OR over independent sources.
supported(S, P, O) :- claims(S, P, O, _).
