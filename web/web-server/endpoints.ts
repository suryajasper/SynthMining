import express from 'express';
import axios from 'axios';

import * as db from './db-functions';

const app = express();

// handle json data
app.use(express.json());

app.post('/createUser', (req, res) => {
  console.log(`--CREATE USER ${req.body.email}`);

  if (!req.body.email || !req.body.firstName || !req.body.lastName || !req.body.password)
    return res.status(400).send({err: 'missing project info'});

  db.createUser({
    name: `${req.body.firstName} ${req.body.lastName}`,
    email: req.body.email,
    password: req.body.password,
  })
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/authenticateUser', (req, res) => {
  console.log(`--AUTH USER ${req.body.email}`);

  if (!req.body.email || !req.body.password)
    return res.status(400).send({err: 'missing authentication info'});

  db.authenticateUser(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/createProject', (req, res) => {
  console.log(`--CREATE PROJECT <${req.body.name}>`);

  if (!req.body.uid)
    return res.status(400).send({err: 'missing user ID'});

  db.createProject(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/updateProject', async (req, res) => {
  console.log(`--UPDATE PROJECT <${req.body.name}>`);

  if (!req.body.uid || !req.body.projectId || !req.body.update) 
    return res.status(400).send({err: 'missing project info'});
  
  db.updateProject(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/removeProject', async (req, res) => {
  console.log(`--REMOVE PROJECT <${req.body.name}>`);

  if (!req.body.uid || !req.body.projectId) 
    return res.status(400).send({err: 'missing project info'});

  db.removeProject(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.get('/getAllProjects', async (req, res) => {
  console.log('--RETRIEVING PROJECTS')

  db.getAllProjects()
    .then(data => res.status(200).json(data))
})

app.get('/getProject', async (req, res) => { 

  console.log(`--RETRIEVING PROJECT INFO from project ${req.query.projectId}`);

  if (!req.query.projectId || !req.query.uid)
    return res.status(400).send({err: 'missing project info'});
  
  let strQuery = {
    uid: req.query.uid.toString(),
    projectId: req.query.projectId.toString(),
  };
  
  db.getProject(strQuery)
    .then(data => res.status(200).json(data))
    .catch(err => res.status(400).send({err: err.toString()}))

})

app.post('/addImages', (req, res) => {
  console.log(`--CREATE IMAGE IDS <${req.body.imgNames?.join(', ')}>`);
  
  if (!req.body.uid || !req.body.projectId || !req.body.imgNames)
    return res.status(400).send({err: 'missing required fields'})

  db.addImages(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/updateImages', async (req, res) => {
  console.log(`--UPDATE IMAGES IN <${req.body.projectId}>`);

  if (!req.body.uid || !req.body.projectId || !req.body.imageIds) 
    return res.status(400).send({err: 'missing required fields'});
  
  db.updateImages(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/removeImages', async (req, res) => {
  console.log(`--REMOVE IMAGES FROM <${req.body.projectId}>`);

  if (!req.body.uid || !req.body.projectId || !req.body.imageIds) 
    return res.status(400).send({err: 'missing required fields'});

  try {
    await db.removeImages(req.body);
    
    await axios.post('http://localhost:2003/removeImages', {
      'img_ids': req.body.imageIds,
    });

    res.status(200).send('success');
  }
  catch (err: unknown) {
    console.log(err);
    res.status(400).send(err);
  }


});

app.post('/createTag', (req, res) => {
  console.log(`--CREATE TAG <${req.body.name || 'unnamed'}> IN <${req.body.projectId}>`);

  if (!req.body.uid || !req.body.projectId)
    return res.status(400).send({err: 'missing required fields'});

  db.createTag(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/updateTag', async (req, res) => {
  console.log(`--UPDATE TAG <${req.body.tagId}> FROM <${req.body.projectId}>`);

  if (!req.body.uid || !req.body.projectId || !req.body.tagId || !req.body.update) 
    return res.status(400).send({err: 'missing required fields'});
  
  db.updateTag(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/removeTag', async (req, res) => {
  console.log(`--REMOVE TAG <${req.body.tagId}> FROM <${req.body.projectId}>`);

  if (!req.body.uid || !req.body.projectId || !req.body.tagId) 
    return res.status(400).send({err: 'missing required fields'});

  db.removeTag(req.body)
    .then(data => res.status(200).send(data))
    .catch(err => res.status(400).send({err: err.toString()}))
});

app.post('/')

export default app;