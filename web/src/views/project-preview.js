import m from 'mithril';
import '../css/projects.scss';

export default class ProjectPreview {
  view(vnode) {
    return m('div.project-preview-container',
      m('div.project-preview-content', {
        onclick: e => {
          m.route.set('/project', { projectId: vnode.attrs._id, })
        }
      }, [

        m('span.project-title', vnode.attrs.name),

        m('div.project-thumbnails-list', (
            vnode.attrs.thumbnails?.length === 4 && 
            vnode.attrs.thumbnails || ['', '', '', '']
          ).map(imgSrc => 
            m('img.project-thumbnail-img', { src: imgSrc, }, )
          ),
        ),

        m('div.project-tags-list', 
          vnode.attrs.tags
            .slice(0, 3)
            .map(tag => 
              m('div.project-tag-item', tag)
            ),
        ),

        m('div.project-description', 
          vnode.attrs.description.substring(0, 200) + '...',
        ),

      ])
    )
  }
}