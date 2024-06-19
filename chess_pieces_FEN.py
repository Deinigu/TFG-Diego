''' Dataset structure for the chess pieces recognition
chess_pieces = {
    0: 'black-king',
    1: 'black-bishop',
    2: 'black-knight',
    3: 'black-rook',
    4: 'black-pawn',
    5: 'black-queen',
    6: 'white-king',
    7: 'white-bishop',
    8: 'white-knight',
    9: 'white-rook',
    10: 'white-pawn',
    11: 'white-queen'
}'''

# FEN notation for chess pieces
chess_pieces = {
    0: 'k',
    1: 'b',
    2: 'n',
    3: 'r',
    4: 'p',
    5: 'q',
    6: 'K',
    7: 'B',
    8: 'N',
    9: 'R',
    10: 'P',
    11: 'Q'
}

# FEM Notation to Unicode chess pieces
chess_pieces_unicode = {
    'K': '♚',
    'B': '♝',
    'N': '♞',
    'R': '♜',
    'P': '♟',
    'Q': '♛',
    'k': '♔',
    'b': '♗',
    'n': '♘',
    'r': '♖',
    'p': '♙',
    'q': '♕'
}