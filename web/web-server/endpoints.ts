import express from 'express';
import User from './models/user';
import Project from './models/project';
import mongoose, { Mongoose } from 'mongoose';

const app = express();

app.get('/', (req, res) => {
  res.send('Hello World');
});

app.post('/createUser', async (req, res) => {
  console.log(req.query);

  console.log(`--CREATE USER ${req.query.email}`);

  const userSetup: mongoose.Document = new User({
    name: `${req.query.firstName} ${req.query.lastName}`,
    email: req.query.email,
    passHash: req.query.password,
  });

  const { _id: uid } = await userSetup.save();

  console.log('--CREATE USER RESULTED SUCCESS');

  return res.status(200).send({ uid });
});

app.post('/authenticateUser', async (req, res) => {
  console.log(`--AUTH USER ${req.query.email}`);

  const indexedUser: any = await User.findOne({ email: req.query.email }).exec();
  if (!indexedUser) {
    // console.log('--AUTH USER FAILED: no user found with email ' + req.query.email);
    return res.status(400).send({err: 'no user found with email'});
  }

  if (indexedUser.validatePassword(req.query.password)) {
    console.log('--AUTH USER FAILED: invalid password');
    return res.status(200).send({uid: indexedUser._id});
  }

  res.status(400).send({err: 'wrong password'});
});

export default app;