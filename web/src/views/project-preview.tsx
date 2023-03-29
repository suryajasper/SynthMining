import React from 'react';
import { withNavigate } from '../utils/react-utils';

import '../css/projects.scss';
import { ProjectBaseAttrs } from './project-loader';
import { NavigateFunction } from 'react-router-dom';

interface ProjectPreviewAttrs extends ProjectBaseAttrs {
  navigate: NavigateFunction;
}

function ProjectPreview(props: ProjectPreviewAttrs) {
  return (
    <div className='project-preview-container'>
      <div 
        className='project-preview-content'
        onClick={() => {
          props.navigate('/project', {  });
        }}
      >

        <span className='project-title'>{props.name}</span>

        <div className='project-thumbnails-list'>{
          ['','','','']
            .map(imgSrc => 
              <img className='project-thumbnail-img' src={imgSrc} />
            )
        }</div>

        <div className='project-tags-list'>{
          props.keywords
            .slice(0, 3)
            .map(tag => 
              <div className='project-tag-item'>{tag}</div>
            )
        }</div>

        <div className='project-description'>
          {props.description.substring(0, 200) + '...'}
        </div>

      </div>
    </div>
  );
}

export default withNavigate(ProjectPreview);