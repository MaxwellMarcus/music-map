def index( data, row=None, col=None ):
    return data.data[ row, : ] if row else data.data[ :, col ]