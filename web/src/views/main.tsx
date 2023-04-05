import React from 'react';

import '../css/main.scss';
import Cookies from '../utils/Cookies';
import { fetchRequest } from '../utils/utils';
import ProjectPreview from './project-preview';
import { ProjectBaseAttrs } from './project-loader';

export default class Main 
  extends React.Component<any, { 
    projects: ProjectBaseAttrs[] 
  }> {
  
  private uid: string | undefined;

  constructor(props: any) {
    super(props);

    this.uid = Cookies.get('uid');
    // if (!this.uid) m.route.set('/login');

    this.state = {
      projects: []
    };

    this.fetchProjects();
  }

  fetchProjects() : void {
    if (!this.uid) return;

    fetchRequest<ProjectBaseAttrs[]>('/getAllProjects', {
      method: 'GET',
      query: { uid: this.uid },
    })
      .then((projects: ProjectBaseAttrs[]) => {
        console.log(projects);
        this.setState({ projects });
      })
  }

  render() {
    return (
      <div className='main-body'>
        <div className='projects-grid'>{
          this.state.projects.map(
            (proj : ProjectBaseAttrs) => <ProjectPreview {...proj} />
          )
        }</div>
      </div>
    );
  }
}