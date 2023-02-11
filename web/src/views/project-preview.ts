import m from 'mithril';
import '../css/projects.scss';
import { ProjectBaseAttrs } from './project-loader';

const ProjectPreview : m.Component<ProjectBaseAttrs> = {
  view({attrs} : m.Vnode<ProjectBaseAttrs>) {
    return m('div.project-preview-container',
      m('div.project-preview-content', {
        onclick: e => {
          m.route.set('/project', { projectId: attrs._id, })
        }
      }, [

        m('span.project-title', attrs.name),

        m('div.project-thumbnails-list', (
            ['', '', '', '']
          ).map(imgSrc => 
            m('img.project-thumbnail-img', { src: imgSrc, }, )
          ),
        ),

        m('div.project-tags-list', 
          attrs.keywords
            .slice(0, 3)
            .map(tag => 
              m('div.project-tag-item', tag)
            ),
        ),

        m('div.project-description', 
          attrs.description.substring(0, 200) + '...',
        ),

      ])
    )
  }
}

export default ProjectPreview;