import hashlib
# function to hide the visible username, ...
# ... not intended for cryptographic security
def hide_username(user):
    hash_object = hashlib.sha1(user.get_username().encode('utf-8'))
    pbHash = hash_object.hexdigest()
    return pbHash#.decode("hex")

def hash256sha(string):
    return hashlib.sha256(string.encode('utf-8')).hexdigest()