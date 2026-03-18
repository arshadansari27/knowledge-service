% Temporal validity rules (placeholder for Phase 1)
expired(S, P, O) :-
    claims(S, P, O, _),
    valid_until(S, P, O, Until),
    current_date(Now),
    Now > Until.
