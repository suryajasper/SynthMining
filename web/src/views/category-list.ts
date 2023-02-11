import m from 'mithril';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { Popup, PopupAttrs } from './popup';
import { icons } from './icons';
import { TagAttrs } from './project-loader';

export interface EditTagPopupAttrs extends PopupAttrs {
  active: boolean;
  data: {
    tag: TagAttrs,
    projectId: string,
    uid: string,
  } | undefined;
}

export class EditTagPopup extends Popup implements m.ClassComponent<EditTagPopupAttrs> {
  private tagId: string | undefined;
  private uid: string | undefined;
  private projectId: string | undefined;

  constructor(vnode: m.CVnode<EditTagPopupAttrs>) {
    super();

    this.actions = [
      {name: 'Save', res: () => this.saveTag(vnode)},
      {name: 'Remove', res: () => this.removeTag(vnode)},
    ];
  }

  onupdate({attrs}: m.CVnodeDOM<EditTagPopupAttrs>): void {
    this.tagId = attrs.data?.tag._id;
    this.projectId = attrs.data?.projectId;
    this.uid = attrs.data?.uid;
  }

  saveTag(vnode: m.CVnode<EditTagPopupAttrs>): void {
    fetchRequest('/updateTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId: this.tagId,
        update: {
          name: this.input['tagName'],
          description: this.input['tagDescription'],
        },
      }
    })
      .then(res => {
        this.hidePopup(vnode);
        vnode.attrs.reloadCallback();
      })
  }

  removeTag(vnode: m.CVnode<EditTagPopupAttrs>): void {
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

  loadPopupContent({attrs}: m.CVnode<EditTagPopupAttrs>): m.Children {
    let tag = attrs.data?.tag;

    this.title = `Edit ${tag?.name}`;
    m.redraw();

    return [
      this.createInputGroup({
        id: 'tagName',
        displayTitle: 'Name',
        initialValue: tag?.name || '',
      }),
      this.createInputGroup({
        id: 'tagDescription',
        displayTitle: 'Description',
        type: 'textarea',
        initialValue: tag?.description || '',
      }),
    ];
  }
}

export interface CategorySelectionsAttrs {
  uid: string | undefined;
  projectId: string;

  categories: TagAttrs[];
  colors: string[];

  addTag() : void;
  res(tag: TagAttrs) : void;
  updatePopupStatus(state: {
    name: string,
    active: boolean,
    data: object,
  }) : void;
}

export class CategorySelections implements m.ClassComponent<CategorySelectionsAttrs> {
  private selected: boolean[];

  constructor(vnode: m.CVnode<CategorySelectionsAttrs>) {
    this.selected = [];
  }

  createCategoryItem(
    vnode: m.CVnode<CategorySelectionsAttrs>, 
    tag: TagAttrs, 
    i: number,
  ) {
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

            vnode.attrs.updatePopupStatus({
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

  view(vnode: m.CVnode<CategorySelectionsAttrs>) {
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