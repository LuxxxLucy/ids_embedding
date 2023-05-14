// Define a grammar called ids
grammar ids;
glyph : 'Glyph' ID ideographic;
ideographic : 
    Part # CJKPART
    | Unknown # Unknown
    | ('<' indicator=Part '>' String* )?  op=IdsBinaryOperator op1=ideographic op2=ideographic # CJKBinaryComposition
    | op=IdsTrinaryOperator op1=ideographic op2=ideographic op3=ideographic #CJKTrinaryComposition
    ;

// IDS_BinaryOperator : [⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻] ;
IdsBinaryOperator : '\u2FF0' | '\u2FF1' | '\u2FF4' | '\u2FF5' | '\u2FF6' | '\u2FF7' | '\u2FF8' | '\u2FF9' | '\u2FFA' | '\u2FFB' ;
// IDS_TrinaryOperator : [⿲⿳] ;
IdsTrinaryOperator : '\u2FF2' | '\u2FF3' ;

// lexical rules
Part : ('\u4E00'..'\u9FFF'
    |'\u3400'..'\u4DBF'
    |'\u{20000}'..'\u{2A6DF}'
    |'\u{2A700}'..'\u{2B73F}'
    |'\u{2B740}'..'\u{2B81F}'
    |'\u{2B820}'..'\u{2CEAF}'
    |'\u{30000}'..'\u{3134F}'
    |'\uF900'..'\uFAFF'
    |'\u{2F800}'..'\u{2FA1F}'
    |'?') ;

Indicator : '\u4E00'..'\u9FFF' ;
Unknown: '?';
ID : ([a-z]|[A-Z]|[0-9])+ ;                 // match lower-case identifiers
String : '('('a'..'z'|'A'..'Z'|' ')+')' ;   // match lower-case identifiers
WS : [ \t\r\n]+ -> skip ;                   // skip spaces, tabs, newlines
