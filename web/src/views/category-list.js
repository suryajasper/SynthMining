import m from 'mithril';
import '../css/tags.scss';
import { initArr } from '../utils/utils';
import { Popup, PopupOuter } from './popup';
import { icons } from './icons';

export class EditTagPopup extends Popup {
  constructor(vnode) {
    super();
  }

  view(vnode) {
    let tag = vnode.attrs.data;

    return m(PopupOuter, {
      active: vnode.attrs.active,
      title: `Edit ${tag?.name}`,
      actions: [
        {name: 'Save', res: console.log},
      ]
    }, [
      this.createInputGroup({
        id: 'tagName',
        displayTitle: 'Name',
      }),
      this.createInputGroup({
        id: 'tagDescription',
        displayTitle: 'Description',
        type: 'textarea',
      }),
    ]);
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
              data: tag,
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