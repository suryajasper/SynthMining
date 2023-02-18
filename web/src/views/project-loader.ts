export interface TagAttrs {
  _id: string,
  projectId: string,
  
  name: string,
  color: string,
  description: string,
  goalQty: number,
  
  highlighted?: boolean,
}

export interface ImageAttrs {
  _id: string,
  name: string,

  projectId: string,
  authorId: string,

  tags: [string],
  validated: boolean,

  src: string,
}

export interface ProjectBaseAttrs {
  _id: string,
  name: string,
  description: string,

  patronId: string,
  keywords: string[],
}

export interface ProjectAttrs extends ProjectBaseAttrs {
  images: ImageAttrs[],
  tags: TagAttrs[],
}