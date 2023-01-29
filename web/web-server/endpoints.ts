import express from 'express';
import multer from 'multer';

import User from './models/user';
import Project from './models/project';
import mongoose, { Mongoose } from 'mongoose';
import fs from 'fs';
import path from 'path';

const app = express();

const upload = multer({
  dest: "./temp/",
});

const TEMP_PATH : string = "./temp/"; 

// handle json data
app.use(express.json());

// handle form data
app.use(upload.array()); 

app.get('/', (req, res) => {
  res.send('Hello World');
});

app.post('/createUser', async (req, res) => {
  console.log(req.body);

  console.log(`--CREATE USER ${req.body.email}`);

  const userSetup: mongoose.Document = new User({
    name: `${req.body.firstName} ${req.body.lastName}`,
    email: req.body.email,
    passHash: req.body.password,
  });

  const { _id: uid } = await userSetup.save();

  console.log('--CREATE USER RESULTED SUCCESS');

  return res.status(200).send({ uid });
});

app.post('/authenticateUser', async (req, res) => {
  console.log(`--AUTH USER ${req.body.email}`);

  const indexedUser: any = await User.findOne({ email: req.body.email }).exec();
  if (!indexedUser) {
    console.log('--AUTH USER FAILED: no user found with email ' + req.body.email);
    return res.status(400).send({err: 'no user found with email'});
  }

  if (indexedUser.validatePassword(req.body.password)) {
    console.log('--AUTH USER SUCCESS');
    return res.status(200).send({uid: indexedUser._id});
  }
  
  console.log('--AUTH USER FAILED: invalid password');
  res.status(400).send({err: 'wrong password'});
});

app.post('/createProject', async (req, res) => {
  console.log(`--CREATE PROJECT <${req.body.name}>`);

  if (!req.body.uid) return res.status(400).send({err: 'missing uid'});

  const projectSetup : mongoose.Document = new Project({
    name: req.body.name,
    description: req.body.name,

    patronId: req.body.uid,
    tags: req.body.tags,

    imgPaths: [],
  });

  const finishedProject : mongoose.Document = await projectSetup.save();

  console.log('--CREATE PROJECT SUCCESS')
  return res.status(200).send({projectId: finishedProject._id});
});

app.post('/updateProject', async (req, res) => {
  console.log(`--UPDATE PROJECT <${req.body.name}>`);

  if (!req.body.uid || !req.body.projectName) 
    return res.status(400).send({err: 'missing project info'});

  await Project.updateOne({
    patronId: req.body.uid,
    name: req.body.projectName,
  }, 
    req.body.update).exec();

  console.log('--UPDATE PROJECT SUCCESS');
  return res.status(200);
});

app.post('/removeProject', async (req, res) => {
  console.log(`--REMOVE PROJECT <${req.body.name}>`);

  if (!req.body.uid || !req.body.projectName) 
    return res.status(400).send({err: 'missing project info'});

  await Project.findOneAndRemove({
    patronId: req.body.uid,
    name: req.body.projectName,
  }).exec();

  console.log('--REMOVE PROJECT SUCCESS');
  return res.status(200);
});

app.get('/getAllProjects', async (req, res) => {
  console.log('--RETRIEVING PROJECTS')
  const allProjects: mongoose.Document[] = await Project.find({}).exec();

  console.log('--RETRIEVING PROJECTS SUCCESS');
  return res.status(200).json(allProjects);
})

app.use((error, req, res, next) => {
  console.log('This is the rejected field ->', error.field);
});

app.post('/uploadImageToProject', upload.single('imgfile'), async (req : any, res) => {
  if (!req.body.projectId || !req.body.uid) return res.sendStatus(400).send('no project id');

  const dirPath : string = `../../imgs/${req.query.projectId}`;

  if (!fs.existsSync(dirPath))
    fs.mkdirSync(dirPath);
  
  let numStored : Number = fs.readdirSync(dirPath).length;

  let newName: string = 
    numStored.toString().padStart(6, '0') + 
    '.' + req.file.originalname.split('.').at(-1);
  
  let tempPath: string = path.join(TEMP_PATH, req.file.originalname);
  let newPath: string = path.join(dirPath, newName);
  
  fs.renameSync(tempPath, newPath);

  const currProj: any = await Project.findById(req.body.projectId).exec();

  await Project.findByIdAndUpdate(req.body.projectId, {
    $push: {
      imgPaths: {
        filePath: newPath,
        authorId: req.body.uid,
        validated: req.body.uid === currProj.authorId,
      }
    }
  }).exec();
  
  res.status(200).send({numStored});

});

app.get('/getProject', async (req, res) => { 

  console.log(`--RETRIEVING PROJECT INFO from project ${req.query.projectId}`);

  if (!req.query.projectId || !req.query.uid)
    return res.status(400).send({err: 'missing project info'});
  
  const project : any = await Project.findOne({ 
    projectId: req.query.projectId,
  }).exec();

  if (!project)
    return res.status(400).send({err: 'project not found'});
  
  return res.status(200).send(project);

})

app.post('/')

export default app;