import mongoose from 'mongoose';

import uniquePasswordValidator from 'mongoose-unique-validator';
import crypto from 'crypto';

const UserSchema: mongoose.Schema = new mongoose.Schema({
  name: String,
  email: {
    type: String,
    match: /\S+@\S+\.\S+/,
    lowercase: true,
    trim: true,
    required: true,
    index: true
  },
  interests: [String],
  passHash: String,
  salt: String,
});

UserSchema.pre('save', function(next) {
  // initialize salt and hash password when a user is created
  this.salt = crypto.randomBytes(14).toString('hex');
  this.passHash = crypto.pbkdf2Sync(this.passHash, this.salt, 50_000, 512, 'sha512').toString('hex');
  
  next();
});

UserSchema.methods.validatePassword = function(pwd)  {
  // validate user's password by comparing generated hash with stored hash
  const hash : crypto.BinaryLike = crypto.pbkdf2Sync(pwd, this.salt, 50_000, 512, 'sha512').toString('hex');
  return this.passHash === hash;
};

const User = mongoose.model('User', UserSchema);

export default User;