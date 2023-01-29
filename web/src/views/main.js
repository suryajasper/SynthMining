import m from 'mithril';
import '../css/main.scss';
import Cookies from '../utils/Cookies';
import { fetchRequest } from '../utils/utils';
import ProjectPreview from './project-preview';

export default class Main {
  constructor(vnode) {
    this.uid = Cookies.get('uid');
    if (!this.uid) m.route.set('/login');

    this.projects = [];

    this.fetchProjects();
  }

  fetchProjects() {
    fetchRequest('/getAllProjects', {
      method: 'GET',
      query: { uid: this.uid },
    })
      .then(projects => {
        console.log(projects);
        this.projects = projects;
        m.redraw();
      })
  }

  view(vnode) {
    return m('div.main-body',
      m('div.projects-grid', 
        this.projects.map(proj => 
          m(ProjectPreview, Object.assign({ uid: this.uid, }, proj))
        ),
      )
    );
  }
}