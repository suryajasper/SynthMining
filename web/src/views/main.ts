import m from 'mithril';
import '../css/main.scss';
import Cookies from '../utils/Cookies';
import { fetchRequest } from '../utils/utils';
import ProjectPreview from './project-preview';
import { ProjectBaseAttrs } from './project-loader';

export default class Main implements m.ClassComponent<any> {
  private uid: string | undefined;
  private projects: ProjectBaseAttrs[];

  constructor(vnode : m.CVnode<any>) {
    this.uid = Cookies.get('uid');
    if (!this.uid) m.route.set('/login');

    this.projects = [];

    this.fetchProjects();
  }

  fetchProjects() : void {
    if (!this.uid) return;

    fetchRequest<Array<ProjectBaseAttrs>>('/getAllProjects', {
      method: 'GET',
      query: { uid: this.uid },
    })
      .then(projects => {
        console.log(projects);
        this.projects = projects;
        m.redraw();
      })
  }

  view(vnode : m.CVnode<any>) {
    return m('div.main-body',
      m('div.projects-grid', 
        this.projects.map(proj => 
          m(ProjectPreview, Object.assign({ uid: this.uid, }, proj))
        ),
      )
    );
  }
}