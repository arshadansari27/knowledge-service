% High-confidence: supported and no value conflicts
high_confidence(S, P, O) :-
    supported(S, P, O),
    \+ value_conflict(S, P, _, _).

% Contested: supported but has conflicting values
contested(S, P, O) :-
    supported(S, P, O),
    value_conflict(S, P, O, _).

% Fact overrides claim when both exist
authoritative(S, P, O) :-
    claims(S, P, O, _),
    claim_type(S, P, O, fact).
