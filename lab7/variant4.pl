% =============================================================================
% Variant 4: English Words to Number
% Convert English words of a written number (up to 1000) into numerical digits
% =============================================================================

:- set_prolog_flag(double_quotes, atom).

% =============================================================================
% Word-to-value mappings
% =============================================================================

% Ones (0-9) 
ones(zero,  0).
ones(one,   1).  ones(two,   2).  ones(three, 3).
ones(four,  4).  ones(five,  5).  ones(six,   6).
ones(seven, 7).  ones(eight, 8).  ones(nine,  9).

% Teens (10-19)
teens(ten,       10).  teens(eleven,    11).  teens(twelve,    12).
teens(thirteen,  13).  teens(fourteen,  14).  teens(fifteen,   15).
teens(sixteen,   16).  teens(seventeen, 17).  teens(eighteen,  18).
teens(nineteen,  19).

% Tens multiples 
tens_word(twenty,  20).  tens_word(thirty,  30).  tens_word(forty,  40).
tens_word(fifty,   50).  tens_word(sixty,   60).  tens_word(seventy,70).
tens_word(eighty,  80).  tens_word(ninety,  90).

% =============================================================================
% Converts input to lowercase, splits on spaces, and creates a clean list of atoms.
% =============================================================================

% string_to_atom_list(+String, -Atoms)
string_to_atom_list(String, Atoms) :-
    string_lower(String, LowerString),            % Make it case-insensitive
    split_string(LowerString, " ", " ", Parts),   % Tokenise on spaces
    include([P]>>(P \= ""), Parts, NonEmpty),     % Drop any empty tokens from extra spaces
    maplist([S, A]>>(atom_string(A, S)), NonEmpty, Atoms).

% =============================================================================
% DCG grammar
% Parses the atom list into a numeric value based on English
% =============================================================================

% --- Top-level rules ---
number(1000) --> [one, thousand].
number(N)    --> hundreds(N).
number(N)    --> below_hundred(N).

% --- Hundreds ---
% e.g. "three hundred and forty five" -> 345
hundreds(N) --> [H], { ones(H, HV) },
                [hundred], [and], below_hundred(R),
                { N is HV * 100 + R }.

% e.g. "five hundred" -> 500
hundreds(N) --> [H], { ones(H, HV) },
                [hundred],
                { N is HV * 100 }.

% Teen: "fourteen" -> 14
below_hundred(N) --> [W], { teens(W, N) }.

% Tens + ones: "forty five" -> 45
below_hundred(N) --> [T], { tens_word(T, TV) },
                     [W], { ones(W, OV) },
                     { N is TV + OV }.

% Tens only: "sixty" -> 60
below_hundred(N) --> [W], { tens_word(W, N) }.

% Single digit: "seven" -> 7
below_hundred(N) --> [W], { ones(W, N) }.

% =============================================================================
% Public predicate
% =============================================================================

% to_num(+WordString, -N)
% Main entry point.
to_num(String, N) :-
    string_to_atom_list(String, Atoms),   % Prep the string
    phrase(number(N), Atoms).             % Parse via DCG