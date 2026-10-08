"""Whitespace-token CFRES parser, equivalent to the original regex parser."""
def events(path):
    with path.open() as stream:
        for line in stream:
            prefix,found,body=line.partition('CFRES ')
            if not found:continue
            stamp,colon,owner=prefix.partition(':')
            if not colon:continue
            parts=body.split()
            if not parts:continue
            fields=dict(token.split('=',1) for token in parts[1:] if '=' in token)
            yield int(stamp),owner.strip().rstrip(':'),parts[0],fields
