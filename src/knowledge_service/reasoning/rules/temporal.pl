% Dates are emitted as integers in YYYYMMDD format for numeric comparison.

% Expired: valid_until has passed
expired(S, P, O) :-
    claims(S, P, O, _),
    valid_until(S, P, O, Until),
    current_date(Now),
    Now > Until.

% Currently valid: has temporal bounds and not expired
currently_valid(S, P, O) :-
    claims(S, P, O, _),
    valid_from(S, P, O, From),
    current_date(Now),
    Now >= From,
    \+ expired(S, P, O).

% Temporal supersedes: newer temporal state replaces older for same S-P
supersedes(S, P, O_new, O_old) :-
    claims(S, P, O_new, _),
    claims(S, P, O_old, _),
    valid_from(S, P, O_new, F1),
    valid_from(S, P, O_old, F2),
    F1 > F2,
    O_new \= O_old.
