import express from 'express';
import User from './models/user';
import Project from './models/project';
import mongoose, { Mongoose } from 'mongoose';

const app = express();

app.get('/', (req, res) => {
  res.send('Hello World');
});

app.post('/createUser', async (req, res) => {
  const userSetup: mongoose.Document = new User({
    name: `${req.body.firstName} ${req.body.lastName}`,
    email: req.body.email,
    password: req.body.password,
  });

  const { _id: uid } = await userSetup.save();

  return res.status(200).send({ uid });
});

app.post('/authenticateUser', async (req, res) => {
  const indexedUser: any = await User.findOne({ email: req.body.email }).exec();
  if (!indexedUser) 
    return res.status(400).send({err: 'no user found with email'});

  if (indexedUser.validatePassword(req.body.password))
    return res.status(200).send({uid: indexedUser._id});

  res.status(400).send({err: 'wrong password'});
});

export default app;