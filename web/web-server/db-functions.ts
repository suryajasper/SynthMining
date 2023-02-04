import User from './models/user';
import Project from './models/project';
import mongoose from 'mongoose';

export async function createUser(params: {
  email: string, 
  name: string, 
  password: string,
}) {
  const {email, name, password} = params;

  if (!email || !name || !password)
    throw new Error('Missing parameters email, name, or password');
  
  const userSetup: mongoose.Document = new User({
    name, email, passHash: password,
  });

  const { _id: uid } = await userSetup.save();

  return { uid };
}

export async function authenticateUser(params: {
  email: string, 
  password: string,
}) {
  const {email, password} = params;

  if (!email || !password)
    throw new Error('Missing email or password');
  
  const indexedUser: any = await User.findOne({ email }).exec();

  if (!indexedUser)
    throw new Error('No user found with email');

  if (indexedUser.validatePassword(password))
    return { uid: indexedUser._id };
}

export async function createProject(params: {
  name: string | undefined, 
  uid: string, 
  tags: [string] | undefined,
}) {
  const {name, uid, tags} = params;

  if (!uid) throw new Error('missing uid');

  const projectSetup : mongoose.Document = new Project({
    name: name || 'Untitled Project',
    description: 'New Project',

    patronId: uid,
    tags: tags || [],

    imgPaths: [],
  });

  const finishedProject : mongoose.Document = await projectSetup.save();

  return {projectId: finishedProject._id};
}

export async function getProject(params: {
  uid: string,
  projectId: string,
}) {
  const project : any = await Project.findById(params.projectId).exec();

  if (!project)
    throw new Error('Project not found');
  
  return project;
}

export async function updateProject(params: {
  projectId: string,
  uid: string,
  update: object,
}) {
  const updateRes = await Project.updateOne({
    _id: params.projectId,
    patronId: params.uid,
  }, 
    params.update).exec();
  
  if (updateRes.modifiedCount > 0)
    return {res: 'success'};

  throw new Error('Could not find project');
}

export async function removeProject(params: {
  uid: string,
  projectId: string,
}) {
  await Project.findOneAndRemove({
    _id: params.projectId,
    patronId: params.uid,
  }).exec();

  return {res: 'success'};
}

export async function getAllProjects() {
  const allProjects: mongoose.Document[] = await Project.find({}).exec();

  return allProjects;
}