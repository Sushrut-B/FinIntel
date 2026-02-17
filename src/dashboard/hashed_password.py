import bcrypt

password = b"your_password_here"
hashed = bcrypt.hashpw(password, bcrypt.gensalt())
print(hashed)
