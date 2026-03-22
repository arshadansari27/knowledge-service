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

% Inverse predicate inference: if A contains B, then B part_of A
inverse_holds(S, P2, O) :-
    claims(O, P1, S, _),
    inverse(P1, P2).

% Multi-source corroboration: claim from 2+ independent sources
corroborated(S, P, O) :-
    claims(S, P, O, Src1),
    claims(S, P, O, Src2),
    Src1 \= Src2.
