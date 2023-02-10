import m from 'mithril';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { Popup } from './popup';
import { icons } from './icons';

export class EditTagPopup extends Popup {
  constructor(vnode) {
    super();

    this.actions = [
      {name: 'Save', res: () => this.saveTag(vnode)},
      {name: 'Remove', res: () => this.removeTag(vnode)},
    ];
  }

  onupdate(vnode) {
    this.tagId = vnode.attrs.data?.tag?._id;
    this.projectId = vnode.attrs.data.projectId;
    this.uid = vnode.attrs.data.uid;
  }

  saveTag(vnode) {
    fetchRequest('/updateTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId: this.tagId,
        update: {
          name: this.input.tagName,
          description: this.input.tagDescription,
        },
      }
    })
      .then(res => {
        this.hidePopup(vnode);
        vnode.attrs.reloadCallback();
      })
  }

  removeTag(vnode) {
    fetchRequest('/removeTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId: this.tagId,
      }
    })
      .then(res => {
        this.hidePopup(vnode);
        vnode.attrs.reloadCallback();
      })
  }

  loadPopupContent(vnode) {
    let {tag} = vnode.attrs.data;

    this.title = `Edit ${tag?.name}`;
    m.redraw();

    return [
      this.createInputGroup({
        id: 'tagName',
        displayTitle: 'Name',
        initialValue: tag?.name,
      }),
      this.createInputGroup({
        id: 'tagDescription',
        displayTitle: 'Description',
        type: 'textarea',
        initialValue: tag?.description,
      }),
    ];
  }
}

export class CategorySelections {
  constructor(vnode) {
    this.selected = [];

    this.updatePopupStatus = vnode.attrs.updatePopupStatus;
  }

  createCategoryItem(vnode, tag, i) {
    return m('div.category-item', {
      class: this.selected[i] ? 'selected' : '',
      onclick: e => {
        if (this.selected[i-1] || !(this.selected[i])) {
          this.selected = initArr(this.selected.length, false);
          this.selected[i] = true;

          vnode.attrs.res(tag);
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
      m('span.category-item-title', tag.name),

      m('div.hover-menu', 
        m('button.tag-edit-button', {
          onclick: e => {
            e.stopPropagation();

            this.updatePopupStatus({
              name: 'editTag', 
              active: true,
              data: {
                projectId: vnode.attrs.projectId,
                uid: vnode.attrs.uid,
                tag,
              },
            });
          }
        }, 
          icons.gear,
        )
      )
    ]);
  }

  view(vnode) {
    if (this.selected.length != vnode.attrs.categories.length)
      this.selected = initArr(vnode.attrs.categories.length, true);
    
    return m('div.category-container', [
      m('div.category-content',
        (vnode.attrs.categories || [])
          .map((cat, i) => 
            this.createCategoryItem(vnode, cat, i)
          )
          .concat([
            m('div.category-item.new-category-button.selected', {
              onclick: vnode.attrs.addTag,
            }, [
              m('span.category-item-new-icon', '+'),
              m('span.category-item-title', 'Add Tag'),
            ])
          ])
      ),
    ]);
  }
}