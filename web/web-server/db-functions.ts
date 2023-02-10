import UserModel from './models/user';
import { ImageModel, TagModel, ProjectModel, } from './models/project';
import mongoose from 'mongoose';

export async function createUser(params: {
  email: string, 
  name: string, 
  password: string,
}) {
  const {email, name, password} = params;

  if (!email || !name || !password)
    throw Error('Missing parameters email, name, or password');
  
  const userSetup: mongoose.Document = new UserModel({
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
    throw Error('Missing email or password');
  
  const indexedUser: any = await UserModel.findOne({ email }).exec();

  if (!indexedUser)
    throw Error('No user found with email');

  if (indexedUser.validatePassword(password))
    return { uid: indexedUser._id };
}

export async function createProject(params: {
  name?: string, 
  description?: string,
  uid: string,
}) {
  const {name, description, uid} = params;

  if (!uid) throw Error('missing uid');

  const projectSetup : mongoose.Document = new ProjectModel({
    name: name || 'Untitled Project',
    description: description || '',
    patronId: uid,
  });

  const finishedProject : mongoose.Document = await projectSetup.save();

  return {projectId: finishedProject._id};
}

export async function getProject(params: {
  uid: string,
  projectId: string,
}) {
  const { uid, projectId } = params;

  let project : object = await ProjectModel.findById(projectId).lean();

  if (!project)
    throw Error('Project not found');

  project = Object.assign(project, {
    images : await ImageModel.find({ projectId }).exec() || [],
    tags   : await TagModel  .find({ projectId }).exec() || [],
  });

  return project;
}

export async function updateProject(params: {
  projectId: string,
  uid: string,
  update: object,
}) {
  const updateRes = await ProjectModel.updateOne(
    {
      _id: params.projectId,
      patronId: params.uid,
    }, 
    params.update, 
    { upsert: false, },
  ).exec();
  
  if (updateRes.modifiedCount > 0)
    return {res: 'success'};

  throw Error('Could not find project');
}

export async function removeProject(params: {
  uid: string,
  projectId: string,
}) {
  await ProjectModel.findOneAndRemove({
    _id: params.projectId,
    patronId: params.uid,
  }).exec();

  return {res: 'success'};
}

export async function addImages(params: {
  projectId: string,
  uid: string, 

  imgNames: string[],
  tags?: string[],
}) {
  if (!params.uid) throw Error('missing uid');
  if (params.imgNames.length === 0) throw Error('missing images');

  const projectDoc : any = await ProjectModel.findById(params.projectId).exec();

  let validated = (projectDoc.patronId == params.uid);

  let imageIds: string[] = [];

  for (let imgName of params.imgNames) {
    const imageSetup : mongoose.Document = new ImageModel({
      name: imgName,

      projectId: params.projectId,
      authorId: params.uid,
  
      tags: params.tags || [],
      validated,
    });
  
    const finishedImage : mongoose.Document = await imageSetup.save();

    imageIds.push(finishedImage._id);
  }

  return { imageIds };
}

export async function updateImages(params: {
  uid: string,
  projectId: string,

  imageIds: string[],

  validate?: boolean,

  removeTags?: string[],
  addTags?: string[],
}) {
  const updateQuery = {
    _id: { $in: params.imageIds, },

    projectId: params.projectId,
    authorId: params.uid,
  }

  let updateData = {}

  if (params.removeTags && params.removeTags.length > 0)
    updateData['$pull'] = { tags: { $in  : params.removeTags || [] } };
  
  if (params.addTags && params.addTags.length > 0)
    updateData['$push'] = { tags: { $each: params.addTags    || [] } };
  
  if (params.validate) updateData['validated'] = true;

  const updateRes = await ImageModel.updateMany(
    updateQuery, 
    updateData,
    { multi: true, upsert: true, }
  ).exec();

  return { updateCount: updateRes.modifiedCount };
}

export async function removeImages(params: {
  uid: string,
  projectId: string,
  imageIds: string[],
}) {
  const deleteRes = await ImageModel.deleteMany({
    _id: { $in: params.imageIds, },

    projectId: params.projectId,
    authorId: params.uid,
  }).exec();

  return { deleteCount: deleteRes.deletedCount };
}

export async function createTag(params: {
  uid: string,
  projectId: string,

  name?: string, 
  description?: string,
  goalQty?: number,
}) {
  if (!params.uid) throw Error('missing uid');
  if (params.goalQty && !Number.isInteger(params.goalQty)) throw Error('invalid goal quantity');

  const projectInfo : any = await ProjectModel.findById(params.projectId).exec();

  if (projectInfo.patronId != params.uid)
    throw Error('unauthorized to create tags on this project');

  const tagSetup : mongoose.Document = new TagModel({
    name: params.name || 'New Tag',
    description: params.description || '',
    projectId: params.projectId,
    goalQty: params.goalQty || 0,
  });

  const finishedTag : mongoose.Document = await tagSetup.save();

  return finishedTag.toObject();
}

export async function updateTag(params: {
  uid: string,
  projectId: string,
  tagId: string,
  update: object,
}) {
  const updateRes = await TagModel.updateOne(
    {
      _id: params.tagId,
      projectId: params.projectId,
    }, 
    params.update, 
    { upsert: false, },
  ).exec();
  
  if (updateRes.modifiedCount > 0)
    return {res: 'success'};

  throw Error('Could not find project');
}

export async function removeTag(params: {
  uid: string,
  tagId: string,
  projectId: string,
}) {
  await TagModel.findOneAndRemove({
    _id: params.tagId,
    proejctId: params.projectId,
  }).exec();

  return {res: 'success'};
}

export async function getAllProjects() {
  const allProjects: mongoose.Document[] = await ProjectModel.find({}).exec();

  return allProjects;
}