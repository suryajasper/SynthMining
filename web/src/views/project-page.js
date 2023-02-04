import m from 'mithril';
import '../css/project-page.scss';
import { fetchRequest } from '../utils/utils';
import Cookies from '../utils/Cookies';
import ImageUpload from './fileupload';
import ImageList from './img-view';
import colorGen from 'iwanthue';

function ProjectInfo(vnode) {
  return {
    view(vnode) {
      return m('div.info-container', [
        m('span.project-title', vnode.attrs.name),

        m('div.project-description-container', [

          m('div.project-tags-list', vnode.attrs.tags.map(
            tag => m('div.project-tag-item', tag)
          )),

            m('div.project-description', vnode.attrs.description),

        ]),
      ]);
    },
  };
}

function initArr(size, val=0) {
  let arr = [];
  for (let i = 0; i < size; i++)
    arr.push(val);
  return arr;
}

class CategorySelections {
  constructor(vnode) {
    this.res = vnode.attrs.res;
    this.selected = [];
  }

  view(vnode) {
    if (this.selected.length != vnode.attrs.categories.length)
      this.selected = initArr(vnode.attrs.categories.length, true);
    
    return m('div.category-container',
      m('div.category-content',
        (vnode.attrs.categories || [])
          .map((cat, i) => 
            m('div.category-item', {
              class: this.selected[i] ? 'selected' : '',
              onclick: e => {
                if (this.selected[i-1] || !(this.selected[i])) {
                  this.selected = initArr(this.selected.length, false);
                  this.selected[i] = true;

                  this.res(cat);
                } else {
                  this.selected = initArr(this.selected.length, true);
                }
              }
            }, [
              m('span.category-item-color', {
                style: {
                  backgroundColor: vnode.attrs.colors[i]
                }
              }),
              m('span.category-item-title', cat),
            ])
          ),
      ),
    );
  }
}

export default class ProjectPage {
  constructor(vnode) {
    this.uid = Cookies.get('uid');
    this.projectId = vnode.attrs.projectId;

    if (!this.uid) m.route.set('/login');
    
    this.editing = false;
    this.info = {
      name: '',
      description: '',

      patronId: '',
      tags: [''],
    }
    this.categories = [];
    this.colors = [];
    this.images = [];

    this.fetchProjectInfo();
    this.fetchImages();
  }

  fetchProjectInfo() {
    fetchRequest('/getProject', {
      method: 'GET',
      query: {
        projectId: this.projectId,
        uid: this.uid,
      },
    })
      .then(projectInfo => {
        this.info = projectInfo;
        m.redraw();
      })
      .catch(console.error)
  }

  fetchImages(cat) {
    console.log('fetching images ', cat);
    let params = {
      'project_id': this.projectId,
      'max_imgs': 60,
    };
    if (cat) params.category = cat;
    m.request('http://localhost:2003/getProjectImages', {
      method: 'GET',
      params,
    }).then(res => {
      if (!res.err && res.images) {
        console.log(res);
        this.images = res.images;
        this.categories = res.categories;
        this.colors = colorGen(this.categories.length);
        m.redraw();
      }
    }).catch(console.error)
  }

  view(vnode) {
    return m('div.project-page', [

      m('div.left-container', [
        m(CategorySelections, {
          categories: this.categories,
          colors: this.colors,
          res: this.fetchImages.bind(this),
        }),
      ]),

      this.images.length === 0 ? 
        m(ImageUpload, {
          active: true,
          uid: this.uid, 
          imgSrcs: this.images,
          status: e => {
            if (!e.err)
              this.fetchImages();
          } 
        }) : 
        m('div.right-container', [
          m(ImageList, {
            images: this.images
          })
        ])

    ]);
  }
}