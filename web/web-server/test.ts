import connectMongoDB from "./mongodb-init";
import * as db from './db-functions';
import { Project, Image, Tag } from "./models/project";

let uid: string = '63d5b34c90cde35fcd2f8aa3';

let loremipsum: string = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Lacus vestibulum sed arcu non odio euismod lacinia. Mi sit amet mauris commodo quis imperdiet. Vel quam elementum pulvinar etiam non quam lacus. Sed libero enim sed faucibus turpis in eu mi. Nec tincidunt praesent semper feugiat nibh sed pulvinar proin. Eleifend quam adipiscing vitae proin sagittis nisl rhoncus mattis rhoncus. Velit dignissim sodales ut eu sem integer vitae. Sed felis eget velit aliquet sagittis id consectetur. Eu tincidunt tortor aliquam nulla. Nibh praesent tristique magna sit amet purus. Pretium aenean pharetra magna ac placerat vestibulum lectus mauris ultrices. Rhoncus dolor purus non enim praesent. Eget aliquet nibh praesent tristique magna sit amet.'
let splitlorem: string[] = loremipsum.split(' ');

function genProjects(num: number) : object[] {
  let projects: object[] = [];

  for (let i = 0; i < num; i++) {
    let startInd: number = Math.floor(Math.random() * splitlorem.length-3);
    let tags: string[] = splitlorem.slice(startInd, startInd + 3);
  
    projects.push({
      name: `Project ${i}`,
      description: loremipsum,
      uid,
    });
  }

  return projects;
}

connectMongoDB().then(async mongodbconnection => {
  

})